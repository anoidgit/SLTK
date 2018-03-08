#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    训练并保存模型
"""
import os
import sys
import pickle
from time import time
import numpy as np
from optparse import OptionParser
from TorchNN.utils import read_pkl, SentenceDataUtil
from TorchNN.utils import is_interactive, parse_int_list
from TorchNN.layers import BiLSTMCRFModel as SLModel

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


op = OptionParser()
op.add_option('-f', dest='features', default=[0], type='str',
              action='callback', callback=parse_int_list, help='使用的特征列数')
op.add_option('--ri', dest='root_idx', default='./data/train_idx', type='str', help='数据索引根目录')
op.add_option('--rv', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--re', dest='root_embed', default='./res/embed', type='str', help='embed根目录')
op.add_option('--ml', dest='max_len', default=50, type='int', help='实例最大长度')
op.add_option('--ds', dest='dev_size', default=0.2, type='float', help='开发集占比')
op.add_option('--lstm', dest='lstm', default=256, type='int', help='LSTM单元数')
op.add_option('--ln', dest='layer_nums', default=1, type='int', help='LSTM层数')
op.add_option('--fd', dest='feature_dim', default=[64], type='str',
              action='callback', callback=parse_int_list, help='输入特征维度')
op.add_option('--dp', dest='dropout', default=0.5, type='float', help='dropout rate')
op.add_option('--lr', dest='learn_rate', default=0.002, type='float', help='learning rate')
op.add_option('--ne', dest='nb_epoch', default=100, type='int', help='迭代次数')
op.add_option('--mp', dest='max_patience', default=5,
              type='int', help='最大耐心值')
op.add_option('--rm', dest='root_model', default='./model/',
              type='str', help='模型根目录')
op.add_option('--bs', dest='batch_size', default=64, type='int', help='batch size')
op.add_option('-g', '--cuda', dest='cuda', action='store_true', default=True, help='是否使用GPU加速')
op.add_option('--nw', dest='nb_work', default=8, type='int', help='加载数据的线程数')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)

# 初始化数据参数
root_idx = opts.root_idx
path_num = os.path.join(root_idx, 'nums.txt')
max_len = opts.max_len
root_voc = opts.root_voc
features = opts.features
feature2id_dict = dict()
for feature_i in opts.features:
    path_f2id = os.path.join(root_voc, 'feature_{0}_2id.pkl'.format(feature_i))
    feature2id_dict[feature_i] = read_pkl(path_f2id)
label2id_dict = read_pkl(os.path.join(root_voc, 'label2id.pkl'))
feature2id_dict['label'] = label2id_dict
has_label = True
dev_size = opts.dev_size
batch_size = opts.batch_size
num_worker = opts.nb_work

# 初始化数据
dataset = SentenceDataUtil(
    path_num, root_idx, max_len, features, feature2id_dict, shuffle=False)
dataset_train, dataset_dev = dataset.split_train_and_dev(dev_size=dev_size)

# 划分训练集和开发集
data_loader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker)
data_loader_dev = DataLoader(
    dataset_dev, batch_size=batch_size, shuffle=False, num_workers=num_worker)


# 初始化模型参数
path_embed = os.path.join(opts.root_embed, 'word2vec.pkl')
pretrained_embed = None
if os.path.exists(path_embed):
    pretrained_embed = read_pkl(path_embed)
feature_size_dict = dict()
for feature_name in feature2id_dict:
    feature_size_dict[feature_name] = len(feature2id_dict[feature_name]) + 1
feature_dim_dict = dict()
for i, feature_name in enumerate(features):
    if i < len(opts.feature_dim):
        feature_dim_dict[feature_name] = opts.feature_dim[i]
    else:
        feature_dim_dict[feature_name] = 32  # default value 32
if pretrained_embed is not None:  # 以预训练向量维度为准
    feature_dim_dict[features[0]] = pretrained_embed.shape[-1]
dropout_rate = opts.dropout
use_cuda = True
kwargs = {'features': features, 'lstm_units': opts.lstm, 'layer_nums': opts.layer_nums,
          'feature_size_dict': feature_size_dict, 'feature_dim_dict': feature_dim_dict,
          'pretrained_embed': pretrained_embed, 'dropout_rate': dropout_rate,
          'max_len': opts.max_len, 'use_cuda': use_cuda}


# 初始化模型
sl_model = SLModel(kwargs)
print(sl_model)
if use_cuda:
    sl_model = sl_model.cuda()
# 自己在这里调整了adam设置，开启了ams
#optimizer = torch.optim.Adam(sl_model.parameters(), lr=opts.learn_rate, amsgrad=True)
optimizer = torch.optim.Adam(sl_model.parameters(), lr=opts.learn_rate)
criterion = torch.nn.NLLLoss(ignore_index=0)

# 训练
t0 = time()
nb_epoch = opts.nb_epoch
max_patience = opts.max_patience
current_patience = 0
root_model = opts.root_model
if not os.path.exists(root_model):
    os.makedirs(root_model)
path_model = os.path.join(root_model, 'sl.model')
best_dev_loss = sys.maxsize
for epoch in range(nb_epoch):
    sys.stdout.write('epoch {0} / {1}: \r'.format(epoch, nb_epoch))
    total_loss, dev_loss = 0., 0.
    sl_model.train()
    current_count = 0
    for i_batch, sample_batched in enumerate(data_loader_train):
        optimizer.zero_grad()
        for feature_name in sample_batched:
            if use_cuda:
                sample_batched[feature_name] = Variable(sample_batched[feature_name]).cuda()
            else:
                sample_batched[feature_name] = Variable(sample_batched[feature_name])
        lstm_feats = sl_model(sample_batched)

        mask = sample_batched[features[0]] > 0
        loss = sl_model.loss(lstm_feats, mask, sample_batched['label'])
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]

        current_count += sample_batched[features[0]].size()[0]
        sys.stdout.write('epoch {0} / {1}: {2} / {3}\r'.format(
            epoch, nb_epoch, current_count, len(dataset_train)))

    sys.stdout.write('epoch {0} / {1}: {2} / {3}\n'.format(
        epoch, nb_epoch, current_count, len(dataset_train)))

    # 计算开发集loss
    sl_model.eval()
    for i_batch, sample_batched in enumerate(data_loader_dev):
        for feature_name in sample_batched:
            if use_cuda:
                sample_batched[feature_name] = Variable(sample_batched[feature_name]).cuda()
            else:
                sample_batched[feature_name] = Variable(sample_batched[feature_name])
        lstm_feats = sl_model(sample_batched)
        mask = sample_batched[features[0]] > 0
        loss = sl_model.loss(lstm_feats, mask, sample_batched['label'])
        dev_loss += loss.data[0]

    total_loss /= float(len(data_loader_train))
    dev_loss /= float(len(data_loader_dev))
    print('\ttrain loss: {:.4f}, dev loss: {:.4f}'.format(total_loss, dev_loss))

    # 根据开发集loss保存模型
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        # 保存模型
        torch.save(sl_model, path_model)
        print('\tmodel has saved to {0}!'.format(path_model))
        current_patience = 0
    else:
        current_patience += 1
        print('\tno improvement, current patience: {0} / {1}'.format(current_patience, max_patience))
        if max_patience <= current_patience:
            print('finished training! (early stopping, max patience: {0})'.format(max_patience))
            break
duration = time() - t0
print('finished training!')
print('done in {:.1f}s!'.format(duration))
