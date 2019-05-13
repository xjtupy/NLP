#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/5/13 10:12
# @Author: Vincent
# @File  : model.py

import tensorflow  as tf

from NLP.seq2seq_basic import config

"""
构建模型
"""


class Seq2SeqModel(object):

    def __init__(self, data_info):
        self.dataInfo = data_info
        self.input_data = None
        self.targets = None
        self.target_sequence_length = None
        self.source_sequence_length = None
        self.train_graph = None
        self.cost = None

    @staticmethod
    def get_input():
        """
        输入tensor定义
        """
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')

        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_sequence_length')
        source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        return inputs, targets, target_sequence_length, max_target_sequence_length, source_sequence_length

    def get_encoder_layer(self, input_data, source_sequence_length):
        """
        构造Encoder层
        :param input_data: 输入tensor
        :param source_sequence_length: 源数据的序列长度
        :return: 编码器最终的状态向量
        """
        # 1、encoder embedding
        encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, self.dataInfo.source_vocab_size,
                                                               config.encoding_embedding_size)

        # 2、RRN cell堆叠
        def get_lstm_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(config.rnn_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for _ in range(config.num_layers)])

        # 3、动态执行
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=source_sequence_length, dtype=tf.float32)

        return encoder_state

    def decoding_layer_train(self, decoder_input, encoder_state, target_sequence_length, max_target_sequence_length):
        """
        构造decoder层——训练
        :param decoder_input: decoder端输入
        :param encoder_state: encoder端编码的状态向量
        :param target_sequence_length: target数据序列中各字串的长度
        :param max_target_sequence_length: target数据序列字串的最大长度
        :return: 返回RNN cell、全连接层等用于predict
        """
        # 1、embedding
        decoder_embeddings = tf.Variable(
            tf.random_uniform([self.dataInfo.target_vocab_size, config.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        # 2、RNN cell堆叠
        def get_lstm_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(config.rnn_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for _ in range(config.num_layers)])

        # 3、全连接层
        output_layer = tf.layers.Dense(self.dataInfo.target_vocab_size,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 4、decoder training
        # 获得helper对象，只在训练的时候用
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=max_target_sequence_length)

        return training_decoder_output, decoder_embeddings, cell, output_layer

    def decoding_layer_predict(self, decoder_embeddings, rnn_cell, output_layer, max_target_sequence_length,
                               encoder_state):
        """
        构造decoder层——预测
        :param decoder_embeddings: 词嵌入矩阵
        :param rnn_cell: 与训练过程相同的RNN cell
        :param output_layer: 与训练过程相同的全连接层
        :param max_target_sequence_length: target数据序列中字串最大长度
        :param encoder_state: encoder端编码的状态向量
        :return: 预测输出
        """
        # 定义batch_size作为feed_dict参数
        input_batch_size = tf.placeholder(tf.int32, (None), name='input_batch_size')

        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([self.dataInfo.target_letter_to_int['<GO>']], dtype=tf.int32),
                               input_batch_size,
                               name='start_tokens')
        end_token = self.dataInfo.target_letter_to_int['<EOS>']
        # 创建helper对象，只在预测时候用
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     end_token)
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(rnn_cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                      impute_finished=True,
                                                                                      maximum_iterations=max_target_sequence_length)
        return predicting_decoder_output

    def seq2seq_model(self, input_data, source_sequence_length, targets, target_sequence_length,
                      max_target_sequence_length):
        """
        构建seq2seq模型
        :param input_data: 输入原始数据
        :param source_sequence_length: 输入原始数据中各序列长度
        :param targets: 目标数据
        :param target_sequence_length: 目标数据中各序列长度
        :param max_target_sequence_length: 目标数据中各序列最大长度
        :return: 训练和预测输出
        """
        # 获取encoder的状态输出
        encoder_state = self.get_encoder_layer(input_data, source_sequence_length)

        # 预处理后的decoder输入
        decoder_input = self.dataInfo.process_decoder_input(targets)

        # 将状态向量与输入传递给decoder
        training_decoder_output, decoder_embeddings, rnn_cell, output_layer = self.decoding_layer_train(
            target_sequence_length,
            max_target_sequence_length,
            encoder_state,
            decoder_input)

        # RNN单元和全连接层与train时一模一样
        predicting_decoder_output = self.decoding_layer_predict(decoder_embeddings,
                                                                rnn_cell,
                                                                output_layer,
                                                                max_target_sequence_length,
                                                                encoder_state)

        return training_decoder_output, predicting_decoder_output

    def create_graph(self):
        """
        构建模型
        """
        self.train_graph = tf.Graph()

        with self.train_graph.as_default():
            # 获得模型输入
            self.input_data, self.targets, self.target_sequence_length, max_target_sequence_length, self.source_sequence_length = self.get_input()

            # 模型训练
            training_decoder_output, predicting_decoder_output = self.seq2seq_model(self.input_data,
                                                                                    self.source_sequence_length,
                                                                                    self.targets,
                                                                                    self.target_sequence_length,
                                                                                    max_target_sequence_length)

            # 该函数用于返回一个跟input一样维度和内容的张量，相当于y=x
            training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

            # 用sequence_mask计算了每个句子的权重，该权重作为参数传入loss函数，主要用来忽略句子中pad部分的loss。如果是对pad以后的句子进行loop，
            # 那么输出权重都是1，不符合我们的要求
            masks = tf.sequence_mask(self.target_sequence_length, max_target_sequence_length, dtype=tf.float32,
                                     name='masks')

            with tf.name_scope("optimization"):
                # Loss function:当我们的输入是不定长的时候，weights参数常常使用上面的masks
                self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(config.learning_rate)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                    grad is not None]
                optimizer.apply_gradients(capped_gradients)

    def train(self):
        self.create_graph()

        # 将数据集分割为train和validation
        train_source = self.dataInfo.source_int[config.batch_size:]
        target_source = self.dataInfo.target_int[config.batch_size:]

        # 留出一个batch进行验证
        valid_source = self.dataInfo.source_int[:config.batch_size]
        valid_target = self.dataInfo.target_int[:config.batch_size]
        (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
            self.dataInfo.get_batches(valid_target, valid_source, self.dataInfo.source_letter_to_int['<PAD>'],
                                      self.dataInfo.target_letter_to_int['<PAD>']))

        display_step = 50  # 每隔50轮输出loss

        checkpoint = "./checkpoint/trained_model.ckpt"
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(1, config.epochs + 1):
                for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                        self.dataInfo.get_batches(target_source, train_source,
                                                  self.dataInfo.source_letter_to_int['<PAD>'],
                                                  self.dataInfo.target_letter_to_int['<PAD>'])):
                    loss = sess.run([self.cost], feed_dict={self.input_data: sources_batch,
                                                            self.targets: targets_batch,
                                                            self.target_sequence_length: targets_lengths,
                                                            self.source_sequence_length: sources_lengths})

                    if batch_i != 0 and batch_i % display_step == 0:
                        # 计算validation loss
                        validation_loss = sess.run(
                            [self.cost],
                            {self.input_data: valid_sources_batch,
                             self.targets: valid_targets_batch,
                             self.target_sequence_length: valid_targets_lengths,
                             self.source_sequence_length: valid_sources_lengths})

                        print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      config.epochs,
                                      batch_i,
                                      len(train_source) // config.batch_size,
                                      loss,
                                      validation_loss[0]))

            # 保存模型
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
            print('Model Trained and Saved')

    def predict(self, input_word):
        sequence_length = [len(sequence) for sequence in input_word]
        text = self.dataInfo.source_to_seq(input_word, sequence_length)

        checkpoint = "./checkpoint/trained_model.ckpt"

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # 加载模型
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)

            input_data = loaded_graph.get_tensor_by_name('inputs:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
            input_bathc_size = loaded_graph.get_tensor_by_name('input_batch_size:0')

            answer_logits = sess.run(logits, {input_data: text,
                                              target_sequence_length: sequence_length,
                                              source_sequence_length: sequence_length,
                                              input_bathc_size: [len(input_word)]})

        pad = self.dataInfo.source_letter_to_int["<PAD>"]

        for i in range(len(input_word)):
            print('原始输入:', input_word[i])

            print('Source')
            print('  Word 编号:    {}'.format([i for i in text[i]]))
            print('  Input Words: {}'.format(" ".join([self.dataInfo.source_int_to_letter[i] for i in text[i]])))

            print('Target')
            print('  Word 编号:       {}'.format([i for i in answer_logits[i] if i != pad]))
            print('  Response Words: {}'.format(
                " ".join([self.dataInfo.target_int_to_letter[i] for i in answer_logits[i] if i != pad])))
            print()
