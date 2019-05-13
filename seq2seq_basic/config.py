#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/13 10:13
# @Author: Vincent
# @File  : config.py
"""
统一定义相关超参数
"""

# 周期
epochs = 60
# 批处理大小
batch_size = 128
# RNN隐层结点数量
rnn_size = 50
# 堆叠的rnn cell数量
num_layers = 2
# 编码器、解码器输入嵌入大小
encoding_embedding_size = 15
decoding_embedding_size = 15
# 学习率
learning_rate = 0.001


# 文件读取路径
source_file_path = './data/letters_source.txt'
target_file_path = './data/letters_target.txt'
