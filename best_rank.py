# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
# import os
import math
import argparse
import sys
import json

from nltk.tokenize import TweetTokenizer

from utils.ConfigLogger import config_logger
from utils.str2bool import str2bool
from utils.Dataset import Dataset
from graph_func import graph_ques_infer,dot_attention,get_sup_query_attention

"""The model class
"""

class Model:

    def __init__(self, use_elmo, use_glove, use_extra_feature, use_bert,encoding_size, pos_emb_size, ner_emb_size, pos_dict_size,
                 ner_dict_size, max_nodes=500, max_query_size=25, glove_dim=300, query_encoding_type='lstm',max_supports=64):
        # # set placeholder for glove feature when glove feature is used
        self.nodes_glove, self.query_glove, self.nodes_ner, self.nodes_pos = None, None, None, None
        self.query_ner, self.query_pos, self.ner_dict_size, self.pos_dict_size = None, None, None, None
        self.use_glove, self.query_encoding_type, self.use_extra_feature = use_glove, query_encoding_type, use_extra_feature
        self.use_elmo = use_elmo
        self.ner_emb_size, self.pos_emb_size = ner_emb_size, pos_emb_size
        self.max_nodes, self.max_query_size = max_nodes, max_query_size
        self.encoding_size = encoding_size
        self.pos_dict_size, self.ner_dict_size = pos_dict_size, ner_dict_size
        self.use_bert = use_bert
        self.max_supports = max_supports

    def modelProcessing(self, query_length, adj, nodes_mask, bmask, nodes_elmo, query_elmo, nodes_glove, query_glove,
                        nodes_ner, nodes_pos, query_ner, query_pos, dropout,nodes_bert, query_bert,
                        context_bert,nodetosup, sup_length):

        query_mask = tf.tile(tf.expand_dims(tf.range(self.max_query_size, dtype=tf.int32), 0),
                             (tf.shape(query_length)[0], 1)) < tf.expand_dims(query_length, -1)
        query_mask = tf.cast(query_mask, tf.float32)

        ## obtain the multi-level feature for both nodes and query
        nodes_compress, query_compress = self.featureLayer(query_length, nodes_elmo, query_elmo, nodes_glove,
                                                           query_glove,
                                                           nodes_ner, nodes_pos, query_ner, query_pos,
                                                           nodes_bert, query_bert
                                                           )
        nodes = nodes_compress * tf.expand_dims(nodes_mask, -1)
        nodes = tf.nn.dropout(nodes, dropout)
        # method 1
        last_hop = self.graph_query_inference(nodes, query_compress, query_mask, adj, hid_units=64, n_heads=4,
                                              node_mask=nodes_mask, dropout=dropout)
        # last_hop = tf.concat([nodes,last_hop],axis=-1)
        predictions = self.add_sentence_rank(last_hop, context_bert, query_compress, query_mask, bmask,
                                             256,nodetosup, dropout)
        # ## method 2
        # last_hop = self.graph_query_infer_add_context(context_bert, nodetosup, nodes, query_compress, query_mask,
        #                                         adj, hid_units=64, n_heads=4,node_mask=nodes_mask, dropout=dropout)
        # predictions = self.outputLayer(last_hop, bmask)
        return predictions

    def featureLayer(self, query_length, nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner, nodes_pos,
                     query_ner, query_pos,nodes_bert, query_bert):
        # compress and flatten query
        with tf.variable_scope('feature_layer', reuse=tf.AUTO_REUSE):
            query_flat, nodes_flat = None, None
            self.use_elmo = False
            if self.use_elmo:
                query_flat = tf.reshape(query_elmo, (-1, self.max_query_size, 3 * 1024))
                nodes_flat = tf.reshape(nodes_elmo, (-1, self.max_nodes, 3 * 1024))
            if self.use_bert:
                if query_flat is None:
                    query_flat,nodes_flat = query_bert, nodes_bert
                else:
                    query_flat = tf.concat((query_flat, query_bert), -1)
                    nodes_flat = tf.concat((nodes_flat, nodes_bert), -1)
            if self.use_glove:
                if query_flat is None:
                    query_flat, nodes_flat = query_glove, nodes_glove
                else:
                    query_flat = tf.concat((query_flat, query_glove), -1)
                    nodes_flat = tf.concat((nodes_flat, nodes_glove), -1)
            query_compress = None
            if self.query_encoding_type == 'lstm':
                lstm_size = self.encoding_size / 2
                query_compress, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=[tf.nn.rnn_cell.LSTMCell(256), tf.nn.rnn_cell.LSTMCell(lstm_size)],
                    cells_bw=[tf.nn.rnn_cell.LSTMCell(256), tf.nn.rnn_cell.LSTMCell(lstm_size)],
                    inputs=query_flat,
                    dtype=tf.float32,
                    sequence_length=query_length
                )
            elif self.query_encoding_type == 'linear':
                query_compress = tf.layers.dense(query_flat, units=self.encoding_size, activation=tf.nn.tanh)
            nodes_compress = tf.layers.dense(nodes_flat, units=self.encoding_size, activation=tf.nn.tanh)

            ## concatenate POS and NER feature with encoded feature
            if self.use_extra_feature:
                ner_embeddings = tf.get_variable('ner_embeddings', [self.ner_dict_size, self.ner_emb_size])
                pos_embeddings = tf.get_variable('pos_embeddings', [self.pos_dict_size, self.pos_emb_size])
                query_ner_emb = tf.nn.embedding_lookup(ner_embeddings, query_ner)
                query_pos_emb = tf.nn.embedding_lookup(pos_embeddings, query_pos)
                nodes_ner_emb = tf.nn.embedding_lookup(ner_embeddings, nodes_ner)
                nodes_pos_emb = tf.nn.embedding_lookup(pos_embeddings, nodes_pos)
                # (batch_size, max_query_length, hidden_size + ner_emb_size + pos_emb_size)
                query_compress = tf.concat((query_compress, query_ner_emb, query_pos_emb), -1)
     
                nodes_compress = tf.concat((nodes_compress, nodes_ner_emb, nodes_pos_emb), -1)
            return nodes_compress, query_compress

    def prog(self, adj, nodes,mask):
            nodes = tf.stack([tf.layers.dense(inputs=nodes, units=nodes.shape[-1]) for _ in range(adj.shape[1])], 1) * tf.expand_dims(tf.expand_dims(mask, -1), 1)
            update = tf.reduce_sum(tf.matmul(adj, nodes), 1) + tf.layers.dense(nodes, units=nodes.shape[-1]) * tf.expand_dims(mask, -1)
            return update

    def graph_query_head(self, inputs, memory, query_mask, bias_mat, hid_units, node_mask, dropout=1.0):
        with tf.name_scope('ques_my_attn'):
            seq = tf.nn.dropout(inputs,dropout)
            seq_fts = tf.layers.conv1d(seq, hid_units, 1, use_bias=False)  # n*d
            memory_ = tf.nn.relu(tf.layers.dense(memory, hid_units, use_bias=False))  # q*d
            outputs = tf.matmul(seq_fts, tf.transpose(memory_, [0, 2, 1])) / (hid_units ** 0.5)  # n*q
            logits = outputs
            last_hop = logits
            for _ in range(hops):
                last_hop = self.prog(bias_mat, last_hop, node_mask)
            att = tf.layers.dense(tf.concat((last_hop, logits), -1), units=logits.shape[-1],activation=tf.nn.sigmoid) * tf.expand_dims(node_mask, -1)
            last_hop = att * tf.nn.tanh(update) + (1 - att) * logits
            last_hop = tf.nn.softmax(last_hop*tf.expand_dims(query_mask, 1))
            res = tf.matmul(last_hop, memory_)
            return res

    def graph_query_inference(self,inputs, memory, query_mask, bias_mat, hid_units, n_heads, node_mask, dropout=1.0):
        attns = []
        for _ in range(n_heads):
            attns.append(self.graph_query_head(inputs, memory, query_mask, bias_mat, hid_units, node_mask=node_mask,
                                            dropout=dropout))
        h_1 = tf.concat(attns, axis=-1)
        h_1 = tf.nn.dropout(h_1, dropout)
        return h_1

    def add_sentence_rank(self,attentionFlowOutput, context_bert,query,query_mask,bmask, hidden,nodetosup,dropout):
        with tf.variable_scope('add_sentence'):
            indices = nodetosup
            atten = self.multi_head(context_bert, query, query_mask, hidden=64, n_heads=4, keep_prob=dropout)
            mention_atten = tf.gather_nd(atten, indices)
            rawPredictions = tf.squeeze(tf.layers.dense(tf.layers.dense(
                attentionFlowOutput, units=256, activation=tf.nn.tanh), units=1), -1)
            mention_rank = tf.squeeze(tf.layers.dense(tf.layers.dense(
                mention_atten, units=256, activation=tf.nn.tanh), units=1), -1)
        with tf.variable_scope('sum'):
            predictions = rawPredictions + mention_rank
            predictions2 = bmask * tf.expand_dims(predictions, 1)
            predictions2 = tf.where(tf.equal(predictions2, 0),
                                    tf.fill(tf.shape(predictions2), -np.inf), predictions2)
            predictions2 = tf.reduce_max(predictions2, -1)
        return predictions2

    def graph_query_head_add_context(self, context_bert, indices, inputs, memory, query_mask, bias_mat,
                                     hid_units, node_mask, dropout=1.0):
        with tf.name_scope('ques_my_attn'):
            seq = tf.nn.dropout(inputs,dropout)
            seq_fts = tf.layers.conv1d(seq, hid_units, 1, use_bias=False)  # n*d
            memory_ = tf.nn.relu(tf.layers.dense(memory, hid_units, use_bias=False))  # q*d
            outputs = tf.matmul(seq_fts, tf.transpose(memory_, [0, 2, 1])) / (hid_units ** 0.5) # n*q

            context_ = tf.nn.relu(tf.layers.dense(context_bert, hid_units, use_bias=False))
            context_att = tf.matmul(context_, tf.transpose(memory_, [0, 2, 1])) / (hid_units ** 0.5)
            context_att = tf.gather_nd(context_att, indices)

            # logits = tf.nn.softmax(outputs*tf.expand_dims(query_mask, 1))
            last_hop = tf.layers.dense(outputs, outputs.shape[-1], use_bias=False) + \
                       tf.layers.dense(context_att, context_att.shape[-1], use_bias=False)
            for _ in range(hops):
                last_hop = self.prog(bias_mat, last_hop, node_mask)
            last_hop = tf.nn.softmax(last_hop*tf.expand_dims(query_mask, 1))
            res = tf.matmul(last_hop, memory_)
            return res

    def graph_query_infer_add_context(self,context_bert, nodetosup, inputs, memory, query_mask,
                                      bias_mat, hid_units, n_heads, node_mask, dropout=1.0):
        attns = []
        for _ in range(n_heads):
            attns.append(self.graph_query_head_add_context(context_bert, nodetosup, inputs, memory,
                                                           query_mask, bias_mat, hid_units, node_mask=node_mask,
                                            dropout=dropout))
        h_1 = tf.concat(attns, axis=-1)
        h_1 = tf.nn.dropout(h_1, dropout)
        return h_1

    def multi_head(self,inputs, memory, query_mask, hidden, n_heads, keep_prob=1.0):
        output = []
        for _ in range(n_heads):
            d_inputs = tf.nn.dropout(inputs, keep_prob)
            d_memory = tf.nn.dropout(memory, keep_prob)
            JX = tf.shape(inputs)[1]
            inputs_ = tf.layers.dense(d_inputs, units=hidden, use_bias=False, activation=tf.nn.relu)
            memory_ = tf.layers.dense(d_memory, units=hidden, use_bias=False, activation=tf.nn.relu)
            outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
            logits = tf.nn.softmax(outputs*tf.expand_dims(query_mask, 1))
            outputs = tf.matmul(logits, memory_)
            # res = tf.concat([inputs, outputs], axis=2)
            output.append(outputs)
        h_1 = tf.concat(output, axis=-1)
        h_1 = tf.nn.dropout(h_1, keep_prob)
        return h_1
"""The optimizer class
"""


class Optimizer:

    def __init__(self, model, use_elmo, use_glove, use_extra_feature, use_bert, pos_dict_size, ner_dict_size, max_nodes=500,
                 max_query_size=25, max_candidates=80, glove_dim=300,max_supports=64,
                 query_encoding_type='lstm', dynamic_change_lr=True, use_multi_gpu=False):
        self.original_learning_rate = tf.placeholder(dtype=tf.float64)
        self.epoch = tf.placeholder(dtype=tf.int32)

        self.nodes_length = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.query_length = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.sup_length = tf.placeholder(shape=(None,), dtype=tf.int32)
        # self.answer_node_mask = tf.placeholder(shape=(None, max_nodes), dtype=tf.float32)
        self.answer_candidates_id = tf.placeholder(shape=(None,), dtype=tf.int64)
        # 2019.5.2
        # self.adj = tf.placeholder(shape=(None, 3, max_nodes, max_nodes), dtype=tf.float32)
        self.adj = tf.placeholder(shape=(None, 3, max_nodes, max_nodes), dtype=tf.float32)

        self.bmask = tf.placeholder(shape=(None, max_candidates, max_nodes), dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)

        self.use_elmo, self.use_glove, self.use_extra_feature,self.use_bert = use_elmo, use_glove, use_extra_feature,use_bert

        # masks
        nodes_mask = tf.tile(tf.expand_dims(tf.range(max_nodes, dtype=tf.int32), 0),
                             (tf.shape(self.nodes_length)[0], 1)) < tf.expand_dims(self.nodes_length, -1)
        self.nodes_mask = tf.cast(nodes_mask, tf.float32)
        self.nodes_elmo,self.query_elmo =None,None
        self.nodes_bert,self.query_bert,self.context_bert = None,None,None
        self.nodetosup = tf.placeholder(shape=(None, max_nodes,2), dtype=tf.int32)
        if use_elmo:
            self.nodes_elmo = tf.placeholder(shape=(None, max_nodes, 3, 1024), dtype=tf.float32)
            self.query_elmo = tf.placeholder(shape=(None, max_query_size, 3, 1024), dtype=tf.float32)
        if use_glove:
            self.nodes_glove = tf.placeholder(shape=(None, max_nodes, glove_dim), dtype=tf.float32)
            self.query_glove = tf.placeholder(shape=(None, max_query_size, glove_dim), dtype=tf.float32)
        self.nodes_ner,self.nodes_pos,self.query_pos,self.query_ner = None,None,None,None
        if use_extra_feature:
            self.nodes_ner = tf.placeholder(shape=(None, max_nodes,), dtype=tf.int32)
            self.nodes_pos = tf.placeholder(shape=(None, max_nodes,), dtype=tf.int32)
            self.query_ner = tf.placeholder(shape=(None, max_query_size,), dtype=tf.int32)
            self.query_pos = tf.placeholder(shape=(None, max_query_size,), dtype=tf.int32)
            self.ner_dict_size = ner_dict_size
            self.pos_dict_size = pos_dict_size
        if use_bert:
            self.nodes_bert = tf.placeholder(shape=(None, max_nodes, 768), dtype=tf.float32)
            self.query_bert = tf.placeholder(shape=(None, max_query_size, 768), dtype=tf.float32)

        self.context_bert = tf.placeholder(shape=(None,max_supports,1024*3+300), dtype=tf.float32)
        self.predictions = model.modelProcessing(self.query_length, self.adj, self.nodes_mask, self.bmask,
                                                 self.nodes_elmo, self.query_elmo, self.nodes_glove, self.query_glove,
                                                 self.nodes_ner, self.nodes_pos,
                                                 self.query_ner, self.query_pos, self.dropout,
                                                 self.nodes_bert, self.query_bert, self.context_bert,self.nodetosup,
                                                 self.sup_length)

        current_lr = self.original_learning_rate
        if dynamic_change_lr:
            current_lr = self.original_learning_rate / (1 + tf.floor(self.epoch / 5))
        if not use_multi_gpu:
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                self.answer_candidates_id,
                self.predictions, reduction=tf.losses.Reduction.NONE)
            self.loss = tf.reduce_mean(cross_entropy)
            self.train_step = tf.train.AdamOptimizer(learning_rate=current_lr).minimize(self.loss)
        else:
            gpu_indices = [0, 1]
            gpu_num = 2
            split_feature = self.splitForMultiGpu(gpu_num)
            count = 0
            opt = tf.train.AdamOptimizer(learning_rate=current_lr)
            tower_grads, losses = [], []
            for i in gpu_indices:
                with tf.device('/gpu:' + str(i)):
                    with tf.name_scope('gpu_' + str(i)):
                        nodes_elmo, query_elmo = None, None
                        if self.use_elmo:
                            nodes_elmo, query_elmo = split_feature['nodes_elmo'][count], split_feature['query_elmo'][
                                count]
                        nodes_glove, query_glove = None, None
                        if self.use_glove:
                            nodes_glove, query_glove = split_feature['nodes_glove'][count], \
                                                       split_feature['query_glove'][count]
                        nodes_ner, nodes_pos, query_ner, query_pos = None, None, None, None
                        if self.use_extra_feature:
                            nodes_ner, nodes_pos = split_feature['nodes_ner'][count], split_feature['nodes_pos'][count]
                            query_ner, query_pos = split_feature['query_ner'][count], split_feature['query_pos'][count]
                        nodes_bert,query_bert,context_bert = None,None,None
                        context_bert = split_feature['context_bert'][count]
                        if self.use_bert:
                            nodes_bert,query_bert,_ = split_feature['nodes_bert'][count],\
                                                                 split_feature['query_bert'][count],\
                                                                 split_feature['context_bert'][count]
                        predictions = model.modelProcessing(split_feature['query_length'][count],
                                                            split_feature['adj'][count],
                                                            split_feature['nodes_mask'][count],
                                                            split_feature['bmask'][count],
                                                            nodes_elmo, query_elmo, nodes_glove, query_glove, nodes_ner,
                                                            nodes_pos, query_ner,
                                                            query_pos, self.dropout,nodes_bert,query_bert,
                                                            context_bert,split_feature['nodetosup'][count],
                                                            split_feature['sup_length'][count])
                        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                            split_feature['answer_candidates_id'][count],
                            predictions, reduction=tf.losses.Reduction.NONE)
                        loss = tf.reduce_mean(cross_entropy)
                        losses.append(loss)
                        tower_grads.append(opt.compute_gradients(loss))
                        count += 1
            self.loss = tf.reduce_mean(losses)
            grad = self.averageGradients(tower_grads)
            self.train_step = opt.apply_gradients(grad)

    """ Average the gradient calculated by different GPUs
    """

    def averageGradients(self, tower_grads):
        average_grad = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grad.append(grad_and_var)
        return average_grad

    """ Split training data for multiple GPUs (here we only support 2 GPU now)
    """

    def splitForMultiGpu(self, gpu_num):

        query_length = tf.split(self.query_length, gpu_num)
        sup_length = tf.split(self.sup_length, gpu_num)
        answer_candidates_id = tf.split(self.answer_candidates_id, gpu_num)
        adj = tf.split(self.adj, gpu_num)
        bmask = tf.split(self.bmask, gpu_num)
        nodes_mask = tf.split(self.nodes_mask, gpu_num)
        nodetosup = tf.split(self.nodetosup,gpu_num)
        nodes_elmo, query_elmo = None, None
        nodes_glove, query_glove = None, None
        nodes_ner, nodes_pos, query_ner, query_pos = None, None, None, None
        nodes_bert, query_bert, context_bert = None, None, None
        if self.use_elmo:
            nodes_elmo = tf.split(self.nodes_elmo, gpu_num)
            query_elmo = tf.split(self.query_elmo, gpu_num)
        if self.use_glove:
            nodes_glove = tf.split(self.nodes_glove, gpu_num)
            query_glove = tf.split(self.query_glove, gpu_num)
        if self.use_extra_feature:
            nodes_ner = tf.split(self.nodes_ner, gpu_num)
            nodes_pos = tf.split(self.nodes_pos, gpu_num)
            query_ner = tf.split(self.query_ner, gpu_num)
            query_pos = tf.split(self.query_pos, gpu_num)
        if self.use_bert:
            nodes_bert = tf.split(self.nodes_bert, gpu_num)
            query_bert = tf.split(self.query_bert, gpu_num)
        context_bert = tf.split(self.context_bert,gpu_num)
        return {'query_length': query_length, 'answer_candidates_id': answer_candidates_id, 'adj': adj, 'bmask': bmask,
                'nodes_elmo': nodes_elmo, 'query_elmo': query_elmo, 'nodes_mask': nodes_mask,
                'nodes_glove': nodes_glove,
                'query_glove': query_glove, 'nodes_ner': nodes_ner, 'nodes_pos': nodes_pos, 'query_ner': query_ner,
                'query_pos': query_pos, 'nodes_bert':nodes_bert,'query_bert':query_bert,'context_bert':context_bert,
                'nodetosup':nodetosup,'sup_length':sup_length}


""" Check whether the preprocessed file existed in current directory
"""


def checkPreprocessFile(file_name, add_query_node):
    preprocess_file_name = file_name
    if add_query_node:
        preprocess_file_name = preprocess_file_name + '.add_query_node'
    if not os.path.isfile('{}.preprocessed.pickle'.format(preprocess_file_name)):
        return preprocess_file_name, False
    return preprocess_file_name, True


def runEvaluationStage(dev_dataset, session, use_elmo, use_glove, use_extra_feature, use_bert, training_info_interval,
                       model=None,
            batch_size=16, save_json=False):
    finished = False
    eval_correct_count = 0
    eval_sample_count = 0
    eval_interval_count = 0
    answer_dict = {}
    while not finished:
        finished, batch = dev_dataset.next_batch(batch_size)
        feed_dict = {optimizer.nodes_length: batch['nodes_length_mb'],
                     optimizer.query_length: batch['query_length_mb'],
                     optimizer.sup_length: batch['sup_length_mb'],
                     optimizer.adj: batch['adj_mb'],
                     optimizer.bmask: batch['bmask_mb'],
                     optimizer.dropout: 1,
                     optimizer.nodetosup : batch['nodetosup']}
        if use_elmo:
            feed_dict[optimizer.nodes_elmo] = batch['nodes_elmo_mb']
            feed_dict[optimizer.query_elmo] = batch['query_elmo_mb']
        if use_glove:
            feed_dict[optimizer.nodes_glove] = batch['nodes_glove_mb']
            feed_dict[optimizer.query_glove] = batch['query_glove_mb']
        if use_extra_feature:
            feed_dict[optimizer.nodes_pos] = batch['nodes_pos_mb']
            feed_dict[optimizer.nodes_ner] = batch['nodes_ner_mb']
            feed_dict[optimizer.query_ner] = batch['query_ner_mb']
            feed_dict[optimizer.query_pos] = batch['query_pos_mb']
        if use_bert:
            feed_dict[optimizer.nodes_bert] = batch['nodes_bert_mb']
            feed_dict[optimizer.query_bert] = batch['query_bert_mb']
        feed_dict[optimizer.context_bert] = batch['context_bert_mb']
        preds = np.argmax(session.run(optimizer.predictions, feed_dict), -1)
        eval_correct_count += (preds == batch['answer_candidates_id_mb']).sum()
        eval_sample_count += len(batch['query_length_mb'])
        eval_interval_count += len(batch['query_length_mb'])
        if eval_interval_count >= training_info_interval:
            logger.info('%s Dev samples has been done, accuracy = %.3f', eval_sample_count,
                eval_correct_count / eval_sample_count)
            eval_interval_count -= training_info_interval
        if save_json:
            for index, id in enumerate(batch['id_mb']):
                answer_dict[id] = preds[index]
    return eval_correct_count / eval_sample_count, answer_dict

def add_attribute_to_collection(model, optimizer):
    tf.add_to_collection('nodes_length', optimizer.nodes_length)
    tf.add_to_collection('query_length', optimizer.query_length)
    tf.add_to_collection('sup_length', optimizer.sup_length)
    tf.add_to_collection('answer_candidates_id', optimizer.answer_candidates_id)
    tf.add_to_collection('adj', optimizer.adj)
    tf.add_to_collection('bmask', optimizer.bmask)
    tf.add_to_collection('train_step', optimizer.train_step)
    tf.add_to_collection('loss', optimizer.loss)
    tf.add_to_collection('predictions', optimizer.predictions)
    # tf.add_to_collection('predictions2', model.predictions2)
    tf.add_to_collection('original_learning_rate', optimizer.original_learning_rate)
    tf.add_to_collection('epoch', optimizer.epoch)
    tf.add_to_collection('use_elmo', optimizer.use_elmo)
    tf.add_to_collection('use_glove', optimizer.use_glove)
    tf.add_to_collection('use_extra_feature', optimizer.use_extra_feature)
    tf.add_to_collection('nodetosup', optimizer.nodetosup)
    tf.add_to_collection('use_bert', optimizer.use_bert)
    if model.use_elmo:
        tf.add_to_collection('nodes_elmo', optimizer.nodes_elmo)
        tf.add_to_collection('query_elmo', optimizer.query_elmo)
    if model.use_glove:
        tf.add_to_collection('nodes_glove', optimizer.nodes_glove)
        tf.add_to_collection('query_glove', optimizer.query_glove)
    if model.use_extra_feature:
        tf.add_to_collection('nodes_pos', optimizer.nodes_pos)
        tf.add_to_collection('nodes_ner', optimizer.nodes_ner)
        tf.add_to_collection('query_pos', optimizer.query_pos)
        tf.add_to_collection('query_ner', optimizer.query_ner)
        tf.add_to_collection('ner_dict_size', optimizer.ner_dict_size)
        tf.add_to_collection('pos_dict_size', optimizer.pos_dict_size)
    if model.use_bert:
        tf.add_to_collection('nodes_bert', optimizer.nodes_bert)
        tf.add_to_collection('query_bert', optimizer.query_bert)
    tf.add_to_collection('context_bert', optimizer.context_bert)

def write_best_accuray_txt(model_path, accuracy):
    file_name = model_path + '/best.txt'
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as f:
        f.write(str(accuracy) + '\n')
    f.close()

def generate_answer_json(answer_dict, in_file):
    with open(in_file, 'r') as f:
        data = json.load(f)
    final_answer_dict = {}
    for d in data:
        final_answer_dict[d['id']] = d['candidates'][answer_dict[d['id']]]
    with open('predictions.json', 'w') as f:
        json.dump(final_answer_dict, f)

class LoadedOptimizer:

    def __init__(self):
        self.train_step = tf.get_collection('train_step')[0]
        self.loss = tf.get_collection('loss')[0]
        self.original_learning_rate = tf.get_collection('original_learning_rate')[0]
        self.epoch = tf.get_collection('epoch')[0]
        self.nodes_length = tf.get_collection('nodes_length')[0]
        self.query_length = tf.get_collection('query_length')[0]
        self.sup_length = tf.get_collection('sup_length')[0]
        self.answer_candidates_id = tf.get_collection('answer_candidates_id')[0]
        self.adj = tf.get_collection('adj')[0]
        self.bmask = tf.get_collection('bmask')[0]
        self.nodetosup = tf.get_collection('nodetosup')[0]
        self.use_elmo = tf.get_collection('use_elmo')[0]
        self.use_glove = tf.get_collection('use_glove')[0]

        self.use_bert = tf.get_collection('use_bert')[0]
        self.use_extra_feature = tf.get_collection('use_extra_feature')[0]
        self.predictions = tf.get_collection('predictions')[0]
        if self.use_elmo:
            self.nodes_elmo = tf.get_collection('nodes_elmo')[0]
            self.query_elmo = tf.get_collection('query_elmo')[0]
        if self.use_glove:
            self.nodes_glove = tf.get_collection('nodes_glove')[0]
            self.query_glove = tf.get_collection('query_glove')[0]
        if self.use_extra_feature:
            self.nodes_pos = tf.get_collection('nodes_pos')[0]
            self.nodes_ner = tf.get_collection('nodes_ner')[0]
            self.query_pos = tf.get_collection('query_pos')[0]
            self.query_ner = tf.get_collection('query_ner')[0]
            self.pos_dict_size = tf.get_collection('pos_dict_size')[0]
            self.ner_dict_size = tf.get_collection('ner_dict_size')[0]
        if self.use_bert:
            self.nodes_bert = tf.get_collection('nodes_bert')[0]
            self.query_bert = tf.get_collection('query_bert')[0]
        self.context_bert = tf.get_collection('context_bert')[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('dev_file', type=str)
    parser.add_argument('--is_masked', type=bool, default=False, help='using masked data or not')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue last train using saved model or' +
            ' start a new training')
    parser.add_argument('--cpu_number', type=int, default=4, help='The number of CPUs used in current script')
    parser.add_argument('--add_query_node', type=bool, default=False, help='Whether the entity in query is involved '+
            'in the construction of graph')
    parser.add_argument('--evaluation_mode', type=bool, default=False, help='Whether using evaluation mode')
    parser.add_argument('--use_elmo', type=str2bool, default=True, help='Whether use the ELMo as a feature for each node')
    parser.add_argument('--use_bert', type=str2bool, default=False,
                        help='Whether use the ELMo as a feature for each node')
    parser.add_argument('--use_glove', type=str2bool, default=True, help='Whether use the GloVe as a feature for each node')
    parser.add_argument('--use_extra_feature', type=str2bool, default=False, help='Whether use extra feature for ' +
            'each node, e.g. NER and POS')
    parser.add_argument('--use_multi_gpu', type=str2bool, default=False, help='Whether use multiple GPUs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--hop_num', type=int, default=5, help='Hop num in GCN layer, in other words the depth of GCN')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch number for the training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--info_interval', type=int, default=1000, help='The interval to display training loss info')
    parser.add_argument('--dropout', type=float, default=0.8, help="Keep rate for dropout in model")
    parser.add_argument('--encoding_size', type=int, default=512, help='The encoding output size for both query and nodes')
    parser.add_argument('--pos_emb_size', type=int, default=8, help='The size of POS embedding')
    parser.add_argument('--ner_emb_size', type=int, default=8, help='The size of NER embedding')

    args = parser.parse_args()
    in_file = args.in_file
    dev_file = args.dev_file
    is_masked = args.is_masked
    continue_train = args.continue_train
    evaluation_mode = args.evaluation_mode
    cpu_number = args.cpu_number
    add_query_node = args.add_query_node
    use_elmo = args.use_elmo
    use_glove = args.use_glove
    use_extra_feature = args.use_extra_feature
    use_bert = args.use_bert
    use_multi_gpu = args.use_multi_gpu
    learning_rate = args.lr
    hops = args.hop_num
    epochs = args.epochs
    batch_size = args.batch_size
    training_info_interval = args.info_interval
    dropout = args.dropout
    encoding_size = args.encoding_size
    pos_emb_size = args.pos_emb_size
    ner_emb_size = args.ner_emb_size

    options_file = 'data/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = 'data/elmo_2x4096_512_2048cnn_2xhighway_weights'
    encoding_type_map = {'lstm':'lstm','linear':'linear'}

    model_name = 'CQ-GCN'
    if evaluation_mode:
        logger = config_logger('evaluation/' + model_name)
    else:
        logger = config_logger('CQ_GCN')

    model_path = os.getcwd() + '/best_models/' + model_name + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenize = TweetTokenizer().tokenize
    logger.info('Hop number is %s', hops)
    logger.info('Learning rate is %s', learning_rate)
    logger.info('Training epoch is %s', epochs)
    logger.info('Batch size is %s', batch_size)
    logger.info('Dropout rate is %f', dropout)
    logger.info('Encoding size for nodes and query feature is %s', encoding_size)
    query_feature_type = encoding_type_map['lstm']
    logger.info('Encoding type for query feature is %s', query_feature_type)
    dynamic_change_learning_rate = True
    logger.info('Is learning rate changing along with epoch count: %s', dynamic_change_learning_rate)

    tf.reset_default_graph()

    train_file_name_prefix, fileExist = checkPreprocessFile(in_file, add_query_node)
    if not fileExist:
        logger.info('Cannot find preprocess data %s, program will shut down.',
            '{}.preprocessed.pickle'.format(train_file_name_prefix))
        sys.exit()
    dev_file_name_prefix, fileExist = checkPreprocessFile(dev_file, add_query_node)
    if not fileExist:
        logger.info('Cannot find preprocess data %s, program will shut down.',
            '{}.preprocessed.pickle'.format(dev_file_name_prefix))
        sys.exit()
    if not evaluation_mode:
        logger.info('Loading preprocessed training data file %s', '{}.preprocessed.pickle'.format(train_file_name_prefix))
        dataset = Dataset(train_file_name_prefix, use_elmo, use_glove, use_extra_feature, max_nodes=500,
                max_query_size=25, max_candidates=80, max_candidates_len=10)
        logger.info('Loading preprocessed development data file %s', '{}.preprocessed.pickle'.format(dev_file_name_prefix))
        dev_dataset = Dataset(dev_file_name_prefix, use_elmo, use_glove, use_extra_feature, max_nodes=500,
                max_query_size=25, max_candidates=80, max_candidates_len=10)
    else:
        logger.info('Loading preprocessed evaluation data file %s',
                '{}.preprocessed.pickle'.format(dev_file_name_prefix))
        dataset = Dataset(dev_file_name_prefix, use_elmo, use_glove, use_extra_feature, max_nodes=500,
            max_query_size=25, max_candidates=80, max_candidates_len=10)

    pos_dict_size, ner_dict_size = 0, 0
    if use_extra_feature:
        pos_dict_size, ner_dict_size = dataset.getPosAndNerDictSize()

    config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': cpu_number}, allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    session = tf.Session(config=config)

    if (not continue_train) and (not evaluation_mode):
        model = Model(use_elmo, use_glove, use_extra_feature, use_bert,encoding_size, pos_emb_size, ner_emb_size, pos_dict_size,
                ner_dict_size)
        optimizer = Optimizer(model, use_elmo, use_glove, use_extra_feature,use_bert, pos_dict_size, ner_dict_size,
                dynamic_change_lr=dynamic_change_learning_rate, use_multi_gpu=use_multi_gpu)
        logger.info('Start a new training')
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        session_op = [optimizer.train_step, optimizer.loss]
        add_attribute_to_collection(model, optimizer)
    else:
        logger.info('Using previously trained model in /models')
        model = Model(use_elmo, use_glove, use_extra_feature, use_bert,encoding_size, pos_emb_size, ner_emb_size, pos_dict_size,
            ner_dict_size)
        optimizer = Optimizer(model, use_elmo, use_glove, use_extra_feature, use_bert, pos_dict_size, ner_dict_size,
            dynamic_change_lr=dynamic_change_learning_rate, use_multi_gpu=use_multi_gpu)
        saver = tf.train.Saver()
        saver.restore(session, './best_models/' + model_name + '/model.ckpt')
        session_op = [optimizer.train_step, optimizer.loss]

    if not evaluation_mode:
        logger.info('=============================')
        logger.info('Starting Training.....')
        best_accuracy = 0
        best_test_accuracy = 0
        for i in range(epochs):
            logger.info('=============================')
            logger.info('Starting Training Epoch %s', i + 1)
            sample_count = 0
            epoch_finished = False
            interval_count = 0
            loss_count = 0
            loss_sum = 0
            problem_index = []
            while not epoch_finished:
                epoch_finished, batch = dataset.next_batch(batch_dim=batch_size, use_multi_gpu=use_multi_gpu)
                feed_dict = {optimizer.nodes_length: batch['nodes_length_mb'],
                             optimizer.query_length: batch['query_length_mb'],
                             optimizer.sup_length:batch['sup_length_mb'],
                                # model.answer_node_mask: batch['answer_node_mask_mb'],
                             optimizer.answer_candidates_id: batch['answer_candidates_id_mb'],
                             optimizer.adj: batch['adj_mb'],
                             optimizer.bmask: batch['bmask_mb'],
                             optimizer.original_learning_rate: learning_rate,
                             optimizer.epoch: i, optimizer.dropout: dropout,
                             optimizer.nodetosup: batch['nodetosup']}
                if use_elmo:
                    feed_dict[optimizer.nodes_elmo] = batch['nodes_elmo_mb']
                    feed_dict[optimizer.query_elmo] = batch['query_elmo_mb']
                if use_glove:
                    feed_dict[optimizer.nodes_glove] = batch['nodes_glove_mb']
                    feed_dict[optimizer.query_glove] = batch['query_glove_mb']
                if use_extra_feature:
                    feed_dict[optimizer.nodes_pos] = batch['nodes_pos_mb']
                    feed_dict[optimizer.nodes_ner] = batch['nodes_ner_mb']
                    feed_dict[optimizer.query_ner] = batch['query_ner_mb']
                    feed_dict[optimizer.query_pos] = batch['query_pos_mb']
                if use_bert:
                    feed_dict[optimizer.nodes_bert] = batch['nodes_bert_mb']
                    feed_dict[optimizer.query_bert] = batch['query_bert_mb']
                feed_dict[optimizer.context_bert] = batch['context_bert_mb']

                _, loss = session.run(session_op, feed_dict=feed_dict)
                sample_count += len(batch['query_length_mb'])
                interval_count += len(batch['query_length_mb'])
                if not math.isinf(loss):
                    loss_sum += loss
                    loss_count += 1
                if interval_count >= training_info_interval:
                    avg_loss = 100
                    if loss_count != 0:
                        avg_loss = loss_sum / loss_count
                    logger.info('%s training samples has been done, loss = %.5f', sample_count, avg_loss)
                    interval_count -= training_info_interval
                    loss_sum = 0
                    loss_count = 0

            logger.info('Epoch %s has been done', i + 1)
            logger.info('-----------------------------')
            logger.info('Running the evaluation stage')
            accuracy, _ = runEvaluationStage(dev_dataset, session, use_elmo, use_glove, use_extra_feature,use_bert,
                    model=model,training_info_interval=training_info_interval)
            if accuracy > best_accuracy:
                logger.info('Evaluation stage finished')
                logger.info('Current model beats the previous best accuracy on dev, previous=%.4f, current=%.4f',
                        best_accuracy, accuracy)
                best_accuracy = accuracy
                saver.save(session, model_path + 'model.ckpt')
                write_best_accuray_txt(model_path, best_accuracy)
                logger.info('Current best model has been saved.')
            else:
                logger.info('Evaluation stage finished')
                logger.info(
                    'Current model did not beat the previous best accuracy on dev, previous=%.3f, current=%.3f',
                    best_accuracy, accuracy)
            logger.info('=============================')
    else:
        logger.info('=============================')
        logger.info('Starting Evaluation.....')
        accuracy, answer_dict = runEvaluationStage(dataset, session, use_elmo, use_glove, use_extra_feature,use_bert,
            model=model, save_json=True,training_info_interval=training_info_interval)
        generate_answer_json(answer_dict, in_file)
        logger.info('Current accuracy=%.5f', accuracy)
        logger.info('Evaluation stage finished')
        logger.info('=============================')
