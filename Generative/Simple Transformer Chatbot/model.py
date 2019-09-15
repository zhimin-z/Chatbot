#encoding=utf-8
import numpy as np
import tensorflow as tf
from utils import load_sor_vocab,load_mub_vocab
from tensorflow.contrib.layers.python.layers import initializers
from modules import *

class Model(object):
    def __init__(self, config,is_train=True):
        self.is_train =  is_train
        self.config = config
        self.lr = config["learning_rate"]
        self.maxlen = config['sequence_length']
        self.dropout_rate = config['dropout_rate']
        self.hidden_units = config['hidden_units']
        self.num_blocks = config['num_blocks']
        self.num_heads = config['num_heads']        
        
        self.global_step = tf.Variable(0,trainable=False)  
        #定义编码输入input
        self.sor_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='sorinput')
        #定义编码输入output
        self.out_inputs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='outinput')
        self.decode_input = tf.concat((tf.ones_like(self.out_inputs[:, :1])*2, self.out_inputs[:, :-1]), -1)
        word2id,id2word = load_sor_vocab()
        # Encoder
        with tf.variable_scope("encoder"):
            self.enc = embedding(self.sor_inputs, len(word2id), self.hidden_units,scale=True,scope="enc_embed")
            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)
            # Positional Encoding
            if False:
                self.enc += positional_encoding(self.sor_inputs,num_units=self.hidden_units,zero_pad=False,scale=False,scope="enc_pe")
            else:
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.sor_inputs)[1]), 0), [tf.shape(self.sor_inputs)[0], 1]),vocab_size=self.maxlen, 
                                      num_units=self.hidden_units,zero_pad=False,scale=False,scope="enc_pe")

            self.enc *= key_masks
            # Dropout
            self.enc = tf.layers.dropout(self.enc,rate=self.dropout_rate,training=tf.convert_to_tensor(self.is_train))   
            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,keys=self.enc,num_units=self.hidden_units,num_heads=self.num_heads,dropout_rate=self.dropout_rate,is_training=self.is_train,
                                                                causality=False)
                    # Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*self.hidden_units, self.hidden_units])  
        #Decode
        with tf.variable_scope("decoder"):
            # Embedding
            self.dec = embedding(self.decode_input,vocab_size=len(word2id),num_units=self.hidden_units,scale=True,scope="dec_embed") 
            key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)), -1)
            # Positional Encoding
            if False:
                self.dec += positional_encoding(self.decode_input,vocab_size=self.maxlen,num_units=self.hidden_units,zero_pad=False,scale=False,scope="dec_pe")
            else:
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decode_input)[1]), 0), [tf.shape(self.decode_input)[0], 1]),vocab_size=self.maxlen,num_units=self.hidden_units, 
                                              zero_pad=False, 
                                              scale=False,
                                              scope="dec_pe")
            self.dec *= key_masks 
            # Dropout
            self.dec = tf.layers.dropout(self.dec,rate=self.dropout_rate,training=tf.convert_to_tensor(self.is_train)) 
            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec,keys=self.dec,num_units=self.hidden_units,num_heads=self.num_heads, dropout_rate=self.dropout_rate,is_training=self.is_train,
                                                                causality=True, 
                                                                scope="self_attention")
                    # Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec,keys=self.enc,num_units=self.hidden_units,num_heads=self.num_heads,dropout_rate=self.dropout_rate,is_training=self.is_train, 
                                                                causality=False,
                                                                scope="vanilla_attention")
                    # Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4*self.hidden_units, self.hidden_units]) 
        # Final linear projection
        self.logits = tf.layers.dense(self.dec, len(word2id))
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.out_inputs, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.out_inputs))*self.istarget)/ (tf.reduce_sum(self.istarget))
        #tf.summary.scalar('acc', self.acc)   
        # Loss
        self.y_smoothed = label_smoothing(tf.one_hot(self.out_inputs, depth=len(word2id)))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
        self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
       
        # 定义优化器
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr)#, beta1=0.9, beta2=0.98, epsilon=1e-8
            grads_vars = self.optimizer.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g,-5,5),v] for g,v in grads_vars]        
            self.train_op = self.optimizer.apply_gradients(capped_grads_vars,self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
    def create_feed_dict(self,is_train,batch):
        if is_train:
            ask,answer = batch
            feed_dict = {
                self.sor_inputs: np.asarray(ask),
                self.out_inputs: np.asarray(answer)
            }
        else:
            ask,_ = batch
            feed_dict = {
            #self.sor_inputs: np.asarray(ask),
            #self.out_inputs:np.zeros((1, len(ask[0])), np.int32)
            }            
        return feed_dict        
    def run_step(self,sess,is_train,batch):
        feed_dict = self.create_feed_dict(is_train,batch)
        if is_train:
            global_step,y_smoothed,loss,logits,preds,_ = sess.run([self.global_step,self.y_smoothed,self.mean_loss,self.logits,self.preds,self.train_op],feed_dict)                 
            return global_step, loss
        else:
            ask,_ = batch
            preds = np.ones((1,20), np.int32)
            #preds[:,0] = 2
            #preds[:,19] = 3
            for i in range(20):
                _preds = sess.run(self.preds, {self.sor_inputs: np.asarray(ask), self.out_inputs:preds})
                preds[:,i] = _preds[:,i]                
            #preds = sess.run([self.preds], feed_dict)
            return preds
    def evaluate_line(self, sess, inputs):
        probs = self.run_step(sess, False, inputs)
        return probs     
        
