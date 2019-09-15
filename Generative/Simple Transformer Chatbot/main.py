#encoding=utf-8
from __future__ import print_function
import tensorflow as tf
import os, codecs,sys
import numpy as np
import pandas as pd
from utils import load_sentences,BatchManager,create_model_and_embedding,get_logger,save_model,input_from_line,load_sor_vocab,load_mub_vocab
from model import Model
from flask import Flask, jsonify, request
from collections import OrderedDict

flags = tf.app.flags
flags.DEFINE_integer("block",6,"layer_size")
flags.DEFINE_integer("sequence_length",20,"word vector dim")
flags.DEFINE_integer("steps_check", 10, "steps per checkpoint")
flags.DEFINE_integer("num_of_epoch", 100000, "epoch number")
flags.DEFINE_integer("batch_size",64 ,"word vector dim")
flags.DEFINE_integer('hidden_units',128,'   ')
flags.DEFINE_integer('num_blocks',6,'   ')
flags.DEFINE_integer('num_heads',8,'   ')
flags.DEFINE_float("dropout_rate", 0.0, "Learning rate")

flags.DEFINE_string("model_path","model/","vocab file path")
flags.DEFINE_string("train_sor_path","data/train.ask.tsv","train file path")
flags.DEFINE_string("train_mub_path","data/train.answer.tsv","train file path")
flags.DEFINE_string("logger_path","logger/train.log","vocab file path")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean('flag',True,' ')
FLAGS = tf.app.flags.FLAGS
app = Flask(__name__)
def config_model():
    config = OrderedDict()
    config["optimizer"] = FLAGS.optimizer
    config["layer_size"] = FLAGS.block
    config["sequence_length"] = FLAGS.sequence_length
    config["batch_size"] = FLAGS.batch_size
    config["hidden_units"] = FLAGS.hidden_units
    config["num_blocks"] = FLAGS.num_blocks
    config["num_heads"] = FLAGS.num_heads
    config["dropout_rate"] = FLAGS.dropout_rate    
    
    config["train_sor_path"] = FLAGS.train_sor_path
    config["train_mub_path"] = FLAGS.train_mub_path
    config["model_path"] = FLAGS.model_path
    config["logger_path"] = FLAGS.logger_path
    config["learning_rate"] = FLAGS.learning_rate
    config['flag'] = FLAGS.flag
    return config
def train():
    #加载训练数据并生成可训练数据
    train_sor_data,train_mub_data = load_sentences(FLAGS.train_sor_path,FLAGS.train_mub_path)
    #将训练数据处理成N批次数据
    train_manager = BatchManager(train_sor_data,train_mub_data, FLAGS.batch_size)
    #设置gpu参数
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    #加载FLAGS参数
    config = config_model()
    logger = get_logger(config["logger_path"])
    #计算批次数
    word2id,id2word = load_sor_vocab() 
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model_and_embedding(sess, Model, FLAGS.model_path, config,True)
        logger.info("start training")
        loss = []  
        with tf.device('/gpu:0'):
            for i in range(FLAGS.num_of_epoch):
                for batch in train_manager.iter_batch(shuffle=True):
                    step,batch_loss = model.run_step(sess,True,batch)
                    loss.append(batch_loss)
                    if step%FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{},chatbot loss:{:>9.6f}".format(iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
                if i%10 == 0:
                    save_model(sess, model, FLAGS.model_path,logger) 
def predict():
    word2id,id2word = load_sor_vocab()   
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    config = config_model() 
    logger = get_logger(config["logger_path"])  
    graph = tf.Graph()
    sess = tf.Session(graph=graph,config=tf_config)
    with graph.as_default():
        sess.run(tf.global_variables_initializer())
        model = create_model_and_embedding(sess, Model, FLAGS.model_path, config,False)
        sys.stdout.write('请输入测试句子：')
        sys.stdout.flush()
        sentences = sys.stdin.readline()
        while True:
            sentences = sentences.replace('\n','')        
            rs = model.evaluate_line(sess,input_from_line(sentences,word2id))
            res = ''.join([id2word[w] for w in rs[0]]).split('</S>')[0].strip()
            print(res)
            print('请输入测试句子：',end='')
            sys.stdout.flush()
            sentences = sys.stdin.readline()            
        print('ok')
def main(_):
    predict()
if __name__ == '__main__':
    tf.app.run(main)