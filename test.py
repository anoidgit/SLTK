#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import sys
import codecs
from time import time
from optparse import OptionParser
from TorchNN.utils import read_pkl, SentenceDataUtil
from TorchNN.utils import is_interactive

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


op = OptionParser()
op.add_option('--ri', '--root_idx', dest='root_idx', default='./data/test_idx', type='str', help='数据索引根目录')
op.add_option('--rv', '--root_voc', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--pm', '--path_model', dest='path_model', default='./model/sl.model',
              type='str', help='模型路径')
op.add_option('--ml', dest='max_len', default=None, type='int', help='实例最大长度')
op.add_option('--bs', '--batch_size', dest='batch_size', default=64, type='int', help='batch size')
op.add_option('-g', '--cuda', dest='cuda', action='store_true', default=True, help='是否使用GPU加速')
op.add_option('--nw', dest='nb_work', default=8, type='int', help='加载数据的线程数')
op.add_option('-o', '--output', dest='output', default='./result.txt',
              type='str', help='预测结果存放路径')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)


# 加载模型
path_model = opts.path_model
sl_model = torch.load(path_model)
sl_model.set_use_cuda(opts.cuda)
if opts.max_len:
    sl_model.max_len = opts.max_len

# 初始化数据参数
root_idx = opts.root_idx
path_num = os.path.join(root_idx, 'nums.txt')
root_voc = opts.root_voc
feature2id_dict = dict()
for feature_i in sl_model.features:
    path_f2id = os.path.join(root_voc, 'feature_{0}_2id.pkl'.format(feature_i))
    feature2id_dict[feature_i] = read_pkl(path_f2id)
label2id_dict = read_pkl(os.path.join(root_voc, 'label2id.pkl'))
has_label = False
batch_size = opts.batch_size
use_cuda = opts.cuda
num_worker = opts.nb_work
path_result = opts.output

t0 = time()

# 初始化数据
dataset = SentenceDataUtil(
    path_num, root_idx, sl_model.max_len, sl_model.features, feature2id_dict, shuffle=False)
dataset_test = dataset.get_all_data()
data_loader_test = DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_worker)

# 测试
label2id_dict_rev = dict()
for k, v in label2id_dict.items():
    label2id_dict_rev[v] = k
label2id_dict_rev[0] = 'O'
file_result = codecs.open(path_result, 'w', encoding='utf-8')
current_count, total_count = 0, len(dataset_test)
for i_batch, sample_batched in enumerate(data_loader_test):
    current_count += sample_batched[sl_model.features[0]].size()[0]
    sys.stdout.write('{0} / {1}\r'.format(current_count ,total_count))
    for feature_name in sample_batched:
        if use_cuda:
            sample_batched[feature_name] = Variable(sample_batched[feature_name]).cuda()
        else:
            sample_batched[feature_name] = Variable(sample_batched[feature_name])
    targets_list = sl_model.predict(sample_batched)
    for targets in targets_list:
        targets = list(map(lambda d: label2id_dict_rev[d], targets))
        file_result.write('{0}\n'.format(' '.join(targets)))
sys.stdout.write('{0} / {1}\n'.format(current_count ,total_count))
file_result.close()
print('done in {:.1f}s!'.format(time()-t0))
