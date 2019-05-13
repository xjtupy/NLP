#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/13 10:12
# @Author: Vincent
# @File  : main.py
from NLP.seq2seq_basic.data_process import DataProcess
from NLP.seq2seq_basic.model import Seq2SeqModel

if __name__ == '__main__':
    data = DataProcess()
    seq2seq = Seq2SeqModel(data)
    seq2seq.train()