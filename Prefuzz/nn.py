import os
import sys
import glob
import math
import socket
import numpy as np
import random
from collections import Counter
from utils import utils
from flow import FlowBuilder

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable

HOST = '127.0.0.1'
PORT = 12012

BATCH_SIZE = 16
HIDDEN_1 = 4096
EPOCHS = 100
SELECT_RATIO = 0.4
SHOWMAP_PATH = './afl-showmap'

bitmap_ec = dict()       # seed - edge coverage 存储种子文件到边覆盖的映射；
label_index = dict()     # edge - index in bitmap 存储边到位图索引的映射；
correspond_dict = dict() # edge - corresponding edges 存储边到“兄弟”边的映射；

# global variables
seed_path = str()
program_execute = str()

logger = utils.init_logger('./log_nn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FuzzDataSet(Dataset):
   #调用construct_bitmap()提取位图样本。
   #位图样本来自种子文件，并使用其覆盖的边缘信息
    def __init__(self, seed_dir, program_execute):
        self.seed_list = glob.glob(os.path.join(seed_dir, '*'))
        self.bitmap = construct_bitmap(self.seed_list, program_execute)
        # in(out) dimension
        self.seed_size = utils.obtain_max_seed_size(seed_dir)
        self.edge_size = self.bitmap.shape[1]

    def __len__(self):
        return(self.bitmap.shape[0])

    def __getitem__(self, idx):
        btflow = vectorize_file(self.seed_list[idx], self.seed_size)
        covers = torch.as_tensor(self.bitmap[idx], dtype=torch.float32)
        return btflow, covers


class FuzzNet(nn.Module):
  #使用 PyTorch 库构建的神经网络模型
  #包含两个线性层，中间有一个ReLU激活函数，输出通过一个Sigmoid函数来得到结果
    def __init__(self, in_dim, hidden_1, out_dim):
        super(FuzzNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_1, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def construct_bitmap(seed_list, program_execute):
    #构造位图
    '''Build the edge coverage bitmap of seeds'''
    global label_index
    edge_list = list()
    # acquire edge coverage of each seed
    #获得种子文件的边缘覆盖信息，用于构造原始位图
    cnt, total = 0, len(seed_list)
    for seed in reversed(seed_list):
        if seed not in bitmap_ec.keys():
            cover_list = utils.acquire_edge(SHOWMAP_PATH, seed, program_execute)
          #utils中调用AFLshowmap来获取种子的边缘覆盖信息：
          #def acquire_edge(showmap_path, seed_path, program_execute):
              #showmap_cmd = f'{showmap_path} -q -o /dev/stdout -m 512 -t 500 ./{program_execute} {seed_path}'
              #try:
                #output = subprocess.check_output(shlex.split(showmap_cmd))
              #except subprocess.CalledProcessError:
                #return list()
              #edge_list = [int(line.split(b':')[0]) for line in output.splitlines()]
              #return edge_list
          
            if len(cover_list):
                bitmap_ec[seed] = cover_list
                edge_list.extend(cover_list)
            else:
                # crash file
                dst_path = os.path.join('./crashes/', os.path.split(seed)[1])
                utils.move_file(seed, dst_path)
                seed_list.remove(seed)
                logger.info('Crash file:' + seed)
        else:
            edge_list.extend(bitmap_ec[seed])
    
        if cnt % 100 == 0:
            print(f'parse seed set {str(round(cnt / total, 5) * 100)[:5]}%', end='\r')
        cnt += 1

    label = [ec[0] for ec in Counter(edge_list).most_common()]
    #统计出现次数最多的边，将它们的顺序作为标签（即，出现次数最多的边的标签是0，第二多的是1，以此类推）。生成label_index字典，可以根据一个边找到对应的标签。
    label_index = dict(zip(label, range(len(label))))
    
    # contrcuct raw bitmap
    bitmap = np.zeros((len(seed_list), len(label)))
    #生成一个初始全部为0的二维数组
    #然后遍历每一个种子文件seed，在对应的行中标记出覆盖的边：bitmap[idx][label_index[edge]] = 1。得到了原始的bitmap
    for idx, seed in enumerate(seed_list):
        if seed not in bitmap_ec.keys():
            continue
        for edge in bitmap_ec[seed]:
            bitmap[idx][label_index[edge]] = 1

    # label reduction
    #完成标签缩减，删除重复列，并更新标签索引
    bitmap, idx_inverse = np.unique(bitmap, axis=1, return_inverse=True)
    #通过numpy的unique函数删除了bitmap中重复的列。
    #参数return_inverse=True意味着返回一个数组，表示原数组每个元素在新数组（唯一化后的数组）的位置。
    # update label index
    for label in label_index.keys():
      #使用idx_inverse更新label_index字典：
      #即，删除重复列后，每个边标签更新为它在新bitmap中的列的位置，即label_index[label] = idx_inverse[raw_idx]。
        raw_idx = label_index[label]
        label_index[label] = idx_inverse[raw_idx]

    logger.info('Bitmap dimension:' + str(bitmap.shape))
    #最终的位图：
    #每一行代表一个种子文件，每一列代表一个边（程序执行中的一个路径），若该种子文件覆盖了该边则对应值为1，否则为0
    return bitmap


def vectorize_file(fname, flength):
    with open(fname, 'rb') as fopen:
        btflow = torch.tensor([bt for bt in bytearray(fopen.read())], dtype=torch.float32) / 255
    # pad sequence
    if flength > len(btflow):
        btflow = F.pad(btflow, (0, flength-len(btflow)), 'constant', 0)
    return btflow


#在Neuzz 的nn.py中,对于边选择机制采用完全随机的算法：
#def select_edges(fuzzData, edge_num):
    # random selection mechanism
    #alter_seeds = list()
    #alter_edges = np.random.choice(fuzzData.edge_size, edge_num)
    #for edge in alter_edges:
        #idx_list = np.where(fuzzData.bitmap[:,edge] == 1)[0]
        #alter_seeds.append(np.random.choice(idx_list, 1, replace=False)[0])
    #interested_indice = zip(alter_edges.tolist(), alter_seeds)
    #return interested_indice

#而在PreFuzz中，0.1的概率使用随机算法，其余0.9使用资源高效的边选择机制：
def select_edges(fuzzData, edge_num):
    # candidate edges
    if np.random.rand() < 0.1:
    #0.1概率使用随机选择算法
        # random selection mechanism
        alter_edges = np.random.choice(fuzzData.edge_size, edge_num)
    else:
        candidate_set = set()
        for edge in label_index.keys():
            if check_select_edge(edge):
            #由算法进行选择更加值得继续探索的边
                candidate_set.add(label_index[edge])
        replace_flag = True if len(candidate_set) < edge_num else False
        alter_edges = np.random.choice(list(candidate_set), edge_num, replace=replace_flag)

    alter_seeds = list()
    for edge in alter_edges:
        idx_list = np.where(fuzzData.bitmap[:,edge] == 1)[0]
        alter_seeds.append(random.choice(idx_list))
    
    interested_indice = zip(alter_edges.tolist(), alter_seeds)
    return interested_indice

def check_select_edge(edge_id):
    #检查给定的边（edge）是否在correspond_dict中。如果不在，说明这个边没有与之关联的其他边（“兄弟边”），那么就返回True，选择这个边
    if edge_id not in correspond_dict.keys():
        return True
        #为什么这种情况返回True：
        #如果一个边在 correspond_dict 字典中没有键，
        #那就意味着它没有对应的“兄弟”边
        #那么可能这个边是在控制流图中的一条单独的路径或者是一个条件判断的分支里只有一条路径
        #对此的理解：
        #没有“兄弟”边，也就没有别的边可以和它争夺覆盖资源，
        #也就避免了过度测试某一特定路径的问题。此外，如果一条边没有任何“兄弟”边，那么意味着它在代码结构中是相对独立的，
        #探索这样的边更可能找出程序中的新行为和潜在错误，因此值得优先探索。
   
    #如果给定的边在correspond_dict中，那么就获取这个边所有的对应边作为correspond_set。如果这个集合是空的，同样返回True，选择这个边。
    correspond_set = correspond_dict[edge_id]
    if len(correspond_set) == 0:
        return True

    #如果correspond_set不为空，那么就遍历这个集合，统计其中已经被探索过的边的数量（cover_cnt）。这里的已探索是指这个边在label_index中  
    cover_cnt = 0
    for ce in correspond_set:
        if ce in label_index.keys():
            cover_cnt += 1
    #计算已探索的边的数量与所有对应边的总数量的比值。如果这个比值高于预设的SELECT_RATIO，则返回False，表示不选择这个边；否则返回True，选择这个边
    if cover_cnt / len(correspond_set) > SELECT_RATIO:
        return False
    return True


def gen_adv(fuzzNet, fuzzData, edge_idx, seed_name):
    #生成一条边对应的梯度
    x = vectorize_file(seed_name, fuzzData.seed_size).to(device)#将种子文件向量化
    x = Variable(x, requires_grad=True)#把向量x包装为一个PyTorch Variable对象，并设置其可以进行梯度计算
    
    y = x
    for layer in list(fuzzNet.layers)[:-1]:
        y = layer(y)

    grads = torch.autograd.grad(y[edge_idx], x)[0]  #在得到网络输出y后，计算出网络输出中的第edge_idx项关于输入x的梯度
    grads = grads.cpu().numpy()
    # sort byte indix desc to the gradients
    #把梯度从设备上转回CPU，并转换为numpy格式的数据，对梯度取绝对值grads_abs，将梯度的序号进行排序并返回索引idx，以及梯度的符号sign
    grads_abs = np.absolute(grads)
    idx = np.argsort(-grads_abs)
    sign = np.sign(grads)[idx]

    return idx, sign, grads_abs


def gen_grads(fuzzNet, fuzzData, grads_num):
    #通过select_edges选出需要探索的边。然后，对这些边调用gen_adv方法生成其关于种子输入的梯度。最后，把梯度的序号、符号和对应的种子名称写入文件
    # edge select strategy
    interested_indice = select_edges(fuzzData, grads_num)
    #写入梯度信息到gradient_info_p文件中
    fopen = open('gradient_info_p', 'w')
    cnt, total = 0, grads_num
    for edge_idx, seed_idx in interested_indice:
        seed_name = fuzzData.seed_list[seed_idx]
        idx, sign, _ = gen_adv(fuzzNet, fuzzData, edge_idx, seed_name)
        idx = [str(ele) for ele in idx]
        sign = [str(int(ele)) for ele in sign]
        fopen.write(','.join(idx) + '|' + ','.join(sign) + '|' + seed_name + '\n')

        if cnt % 10 == 0:
            print(f'generate gradients {str(round(cnt / total, 5) * 100)[:5]}%', end='\r')
        cnt += 1

    logger.info('Gradients number:' + str(grads_num))
    fopen.close()


def gen_grads_havoc(fuzzNet, fuzzData, grads_num):
    # edge select strategy
    interested_indice = select_edges(fuzzData, grads_num)

    fopen = open('gradient_info_havoc_p', 'w')
    cnt, total = 0, grads_num
    for edge_idx, seed_idx in interested_indice:
        seed_name = fuzzData.seed_list[seed_idx]
        _, _, grads = gen_adv(fuzzNet, fuzzData, edge_idx, seed_name)
        grads = [str(int(ele * 10000)) for ele in grads]
        fopen.write(','.join(grads) + '|' + seed_name + '\n')

        if cnt % 10 == 0:
            print(f'generate gradients for havoc {str(round(cnt / total, 5) * 100)[:5]}%', end='\r')
        cnt += 1

    logger.info('Gradients number for havoc:' + str(grads_num))
    fopen.close()


def accuracy(y_pred, y_true):
    '''Evaluation function'''
    y_true = y_true.int()
    y_pred = y_pred.round().int()

    edge_num = y_true.numel()
    false_num = edge_num - torch.sum(torch.eq(y_pred, y_true)).item()
    true_one_num = torch.sum(y_true & y_pred).item()

    return true_one_num / (true_one_num + false_num)


def step_decay(epoch):
    drop = 0.7
    epochs_drop = 10.0
    lr_lambda = math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr_lambda


def collate_fn(train_data):
    # sort by file length
    train_data.sort(key=lambda data: len(data[0]), reverse=True)
    train_x, train_y = map(list, zip(*train_data))
    data_len = [len(data) for data in train_x]

    train_x = nn.utils.rnn.pad_sequence(train_x, batch_first=True, padding_value=0)
    train_y = torch.cat(train_y).reshape(len(train_y), -1)

    return train_x.unsqueeze(-1), train_y, data_len


def train_model(fuzzNet, fuzzData, epochs):
    fuzzIter = DataLoader(fuzzData, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(fuzzNet.parameters(), lr=0.0001)
    schedular = LambdaLR(optimizer, lr_lambda=step_decay)

    for epoch in range(epochs):
        loss_sum = 0.0
        acc_sum = 0.0

        for step, (btflow, covers) in enumerate(fuzzIter, 1):
            btflow = btflow.to(device)
            covers = covers.to(device)

            preds = fuzzNet(btflow)
            loss = F.binary_cross_entropy(preds, covers)
            acc = accuracy(preds, covers)

            loss_sum += loss.item()
            acc_sum += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        schedular.step()

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(
            epoch+1, epochs, loss_sum/step, acc_sum/step
        ))
    # save model
    # torch.save(fuzzNet.state_dict(), './fuzz_model.pth')


def nn_lop(grads_num, grads_havoc_num):
    # build model
    fuzzData = FuzzDataSet(seed_path, program_execute)
    fuzzNet = FuzzNet(fuzzData.seed_size, HIDDEN_1, fuzzData.edge_size).to(device)
    logger.info(f'Input dim:{fuzzData.seed_size}\t Out dim:{fuzzData.edge_size}')
    # train model
    train_model(fuzzNet, fuzzData, EPOCHS) 
    # fuzzNet.load_state_dict(torch.load('./fuzz_model.pth'))
    
    # generate gradient values
    gen_grads(fuzzNet, fuzzData, grads_num)
    gen_grads_havoc(fuzzNet, fuzzData, grads_havoc_num)
    logger.info('End of one NN loop')


def init_env(program_path):
    global correspond_dict
    os.path.isdir("./vari_seeds/")  or  os.makedirs("./vari_seeds")
    os.path.isdir("./havoc_seeds/") or  os.makedirs("./havoc_seeds")
    os.path.isdir("./crashes/")     or  os.makedirs("./crashes")

    # construct edge corresponding dict
    logger.info(f'Construct the control-flow')
    flow = FlowBuilder(program_path)
    with open(flow.correspond_target, 'r') as fopen:
        correspond_dict = eval(fopen.readline())
    # initial gradients
    nn_lop(50, 100)


def setup_server():
    global seed_path
    global program_execute

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    logger.info('Server set up, wait for connection...')
    conn, addr = sock.accept()
    logger.info('Connected by fuzzing module:' + str(addr))

    seed_path = conn.recv(1024).decode()
    if not os.path.isdir(seed_path):
        logger.info('Invalid seed folder path:' + str(seed_path))
        sys.exit(-1)
    program_execute = ' '.join(sys.argv[1:])
    
    # initial
    init_env(sys.argv[1])
    conn.sendall(b'start')

    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            nn_lop(100, 2000)
            conn.sendall(b'start')
    conn.close()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python nn.py <target_program_path> <program_arg>')
        sys.exit(-1)
    # Server set
    setup_server()
