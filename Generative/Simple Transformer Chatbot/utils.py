#encoding=utf-8
import codecs
import csv
import array
import numpy as np
import tensorflow as tf
import re
import math
import random
import jieba
import logging
import os
def create_model_and_embedding(session,Model_class,path,config,is_train):
    model = Model_class(config,is_train)
    ckpt = tf.train.get_checkpoint_state(path) 
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())
    return model 
def save_model(sess, model, path,logger):
    checkpoint_path = os.path.join(path, "chatbot.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")
def load_sor_vocab():
    vocab = [line.split()[0] for line in codecs.open('data/vocab.tsv', 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word    
def load_mub_vocab():   
    vocab = [line.split()[0] for line in codecs.open('data/vocab.answer.tsv', 'r', 'utf-8').read().splitlines()]
    #word2idx = {word: idx for idx, word in enumerate(vocab)}
    #idx2word = {idx: word for idx, word in enumerate(vocab)}
    #return word2idx, idx2word    
def load_sentences(sor_path,mub_path):
    de_sents = [line.strip().replace('\r','') for line in codecs.open(sor_path, 'r', 'utf-8').read().split("\n")]
    en_sents = [line.strip().replace('\r','') for line in codecs.open(mub_path, 'r', 'utf-8').read().split("\n")]
    de_sents = [' '.join([i for i in line.strip()])  for line in de_sents]
    en_sents = [' '.join([i for i in line.strip()])  for line in en_sents]
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y 
def create_data(source_sents, target_sents):
    word2id,id2word = load_sor_vocab()
    #mub2id,id2mud = load_mub_vocab()
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [word2id.get(word, 1) for word in (source_sent).split()] # 1: OOV, </S>: End of Text
        y = [word2id.get(word, 1) for word in (target_sent+" </S>").split()] 
        if max(len(x), len(y)) <= 20:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    return x_list, y_list, Sources, Targets 
#实例化日志类
def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
def input_from_line(line, char_to_id):
    inputs = list()
    #把空格替换为$
    line = line.replace(" ", "")    
    #查字典，把输入字符中能查到字典的字符转换为ID值，查不到的字标记为<UNK>
    ids = [char_to_id[char] if char in char_to_id else char_to_id["<UNK>"] for char in line] 
    #+[char_to_id['</S>']]
    inputs.append([ids])
    inputs.append([line])
    return inputs
class BatchManager(object):
    def __init__(self, sor_data,mub_data,batch_size):
        self.batch_data = self.sort_and_pad(sor_data,mub_data,batch_size)
        self.len_data = len(self.batch_data)
    def sort_and_pad(self,sor_data,mub_data, batch_size):
        alldata = []
        for ask,answer in zip(sor_data, mub_data):
            sentence = []
            sentence.append(ask)
            sentence.append(answer)
            alldata.append(sentence)
        num_batch = int(math.ceil(len(alldata) /batch_size))
        
        #sorted_data = sorted(sor_data, key=lambda x: len(x[0]))
        #sorted_data = sor_data
               
        random.shuffle(alldata)
        batch_data = []
        for i in range(num_batch):
            batch_data.append(self.pad_data(alldata[i*int(batch_size) : (i+1)*int(batch_size)]))
        return batch_data
    @staticmethod
    def pad_data(data):
        ask,answer = [],[]
        max_sor = max([len(sentence[0]) for sentence in data])
        max_mub = max([len(sentence[1]) for sentence in data])
        for line in data:
            qpadding = [0] * (max_sor- len(line[0]))
            ask.append(list(line[0])+qpadding)
            apadding = [0] * (max_mub - len(line[1]))
            answer.append(list(line[1])+apadding)            
        return [ask,answer]
    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]