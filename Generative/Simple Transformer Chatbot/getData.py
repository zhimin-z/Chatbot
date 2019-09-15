#enconding=utf-8
import os,sys,csv
import numpy as np
import pandas as pd
import codecs
import tensorflow as tf
from modules import *

def full_to_half(s):
    """
    将全角字符转换为半角字符 
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)

def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "")
    s = s.replace("&rdquo;", "")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)
def setdata(line):
    line = line.replace('。','')
    line = line.replace('？','')
    line = line.replace('！','')
    line = line.replace('，','')
    line = line.replace('.','')
    line = line.replace(',','')
    line = line.replace('?','')
    line = line.replace('!','')
    line = line.replace('“','')
    line = line.replace('”','')
    return line
'''
y = tf.constant([[4,2,3,4,5,6,7,8,9]])
enc = embedding(y, 
                         vocab_size=20, 
                                  num_units=8, 
                                  scale=True,
                                  scope="enc_embed")

key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(enc), axis=-1)), -1)
with tf.Session() as sess:
    initall = tf.global_variables_initializer()
    sess.run(initall)
    print(sess.run(key_masks))
'''
vocab = {line.split()[0]:int(line.split()[1]) for line in codecs.open('data/vocab.tsv', 'r', 'utf-8').read().splitlines()}
fp = codecs.open('data/train.answer.tsv','r',encoding='utf-8-sig').read().split('\n')
#vocab = {}
for w in fp:
    for i in w.strip():
        if i in vocab.keys():
            vocab[i] += 1
        else:
            vocab[i] = 1

with open('data/vocab.tsv','w',encoding='utf-8') as fa:
    for k,v in vocab.items():
        strs = k+' '+str(v)
        fa.write(strs+'\n')
fa.close()
'''
fp = codecs.open('data/xiaohuangji50w_nofenci.conv','r',encoding='utf-8')
i = 1
asks = []
answers = []
sentence = []
for k,w in enumerate(fp):
    w = w.strip()
    if k > 0:
        if "M" not in w and w != 'E':
            continue        
        if i%3 == 0:
            sentence[1] = sentence[1].replace(' ','')
            sentence[2] = sentence[2].replace(' ','')
            if sentence[1][1:] != '' and sentence[2][1:] != '':
                asks.append(sentence[1][1:])
                answers.append(sentence[2][1:])
            sentence = []
            i = 1
            sentence.append(w)
        else:
            i += 1
            sentence.append(w)
    else:
        sentence.append(w)
asks = list(filter(None,asks))
answers = list(filter(None,answers))
'''
fp = codecs.open('data/123.txt','r',encoding='utf-8-sig')
i = 1
asks = []
answers = []
for k,w in enumerate(fp):
    w = w.strip()
    w = full_to_half(w)
    w = replace_html(w)    
    w = setdata(w)
    if k%2 == 0:
        asks.append(w)
    else:
        answers.append(w)
with open('data/train.ask.tsv','w',encoding='utf-8') as fa:
    for w in asks:
        fa.write(w+'\n')
with open('data/train.answer.tsv','w',encoding='utf-8') as fs:
    for w in answers:
        fs.write(w+'\n')
fa.close()
fs.close()
print('ok')
