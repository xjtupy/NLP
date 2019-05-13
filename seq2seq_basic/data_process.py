#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/13 10:12
# @Author: Vincent
# @File  : data_process.py
"""
数据预处理
"""
import tensorflow as tf
import numpy as np
from NLP.seq2seq_basic import config


class DataProcess(object):

    def __init__(self):
        # 源文件数据
        self.source_data = None
        self.target_data = None

        # 源文件数据映射
        self.source_int_to_letter = None
        self.source_letter_to_int = None
        self.target_int_to_letter = None
        self.target_letter_to_int = None

        # 源文件数据转换
        self.source_int = None
        self.target_int = None

        # 词汇表大小
        self.source_vocab_size = None
        self.target_vocab_size = None

        # 数据预处理
        self.read_data()
        self.data_convert()

    def read_data(self):
        with open(config.source_file_path, 'r', encoding='UTF-8') as f:
            self.source_data = f.read().split('\n')

        with open(config.target_file_path, 'r', encoding='UTF-8') as f:
            self.target_data = f.read().split('\n')

    @staticmethod
    def extract_character_vocab(data):
        """
        构造映射表
        """
        # 四种特殊字符
        # < PAD >: 补全字符。
        # < EOS >: 解码器端的句子结束标识符。
        # < UNK >: 替代低频词或者一些未遇到过的词等。
        # < GO >: 解码器端的句子起始标识符。
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

        set_words = list(set([character for line in data for character in line]))

        # 添加特殊字符到映射表
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

        return int_to_vocab, vocab_to_int

    def data_convert(self):
        """
        将文件中的数据转换为计算机能识别的数字
        """
        # 构造源文件映射表
        self.source_int_to_letter, self.source_letter_to_int = self.extract_character_vocab(self.source_data)
        self.target_int_to_letter, self.target_letter_to_int = self.extract_character_vocab(self.target_data)

        # 对源文件数据进行转换
        self.source_int = [[self.source_letter_to_int.get(c, self.source_letter_to_int['<UNK>']) for c in line] for line
                           in self.source_data]
        self.target_int = [[self.target_letter_to_int.get(c, self.target_letter_to_int['<UNK>']) for c in line] + [
            self.target_letter_to_int['<EOS>']] for line in self.target_data]

        # 词汇表大小
        self.source_vocab_size = len(self.source_int_to_letter)
        self.target_vocab_size = len(self.target_int_to_letter)

    def process_decoder_input(self, data):
        """
        处理decoder层输入数据
        在每条target数据前补充<GO>（预测时没有输入语句，在第一个时间步，模型输入<GO>生成第一个字符，接着以第一个字符输入生成第二个字符，直到最后一个字符），
        并移除最后一个字符（最后一个字符预测输出的是<EOS>,我们并不需要）
        :param data:
        :return:
        """
        # 去掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [config.batch_size, -1], [1, 1])
        # 在每条target数据前补充<GO>
        decoder_input = tf.concat([tf.fill([config.batch_size, 1], self.target_letter_to_int['<GO>']), ending], axis=1)

        return decoder_input

    def pad_sentence_batch(self, sentence_batch, pad_int):
        """
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length，
        因为在每个batch中RNN是无法接收长度不一的序列的，不同batch序列长度可以不一样
        :param sentence_batch:
        :param pad_int: <PAD>对应索引号
        :return:
        """
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def get_batches(self, targets, sources, pad_source_int, pad_target_int):
        """
        获取一个batch_size大小的数据集
        :param targets:
        :param sources:
        :param pad_source_int:
        :param pad_target_int:
        :return:
        """
        for batch_i in range(0, len(sources) // config.batch_size):
            start_i = batch_i * config.batch_size
            sources_batch = sources[start_i:start_i + config.batch_size]
            targets_batch = targets[start_i:start_i + config.batch_size]

            # 记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))

            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))

            # 补全序列
            pad_source_batch = np.array(self.pad_sentence_batch(sources_batch, pad_source_int))
            pad_target_batch = np.array(self.pad_sentence_batch(targets_batch, pad_target_int))

            yield pad_target_batch, pad_source_batch, targets_lengths, source_lengths

    def source_to_seq(self, text, sequence_length):
        """
        对源数据进行转换
        """
        sequence_length = max(sequence_length)
        return [[self.source_letter_to_int.get(word, self.source_letter_to_int['<UNK>']) for word in sequence] + [
            self.source_letter_to_int['<PAD>']] * (sequence_length - len(sequence)) for sequence in text]
