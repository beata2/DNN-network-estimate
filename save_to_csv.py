'''
第一层保存至csv
'''

import method as md
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import copy
sess = tf.Session()
# import dill


file_name = "./m/Unit2-model-200"
# arfa = 1.
N_CSI_len = 20   #CSI长度
K_len = 30  #数据帧长
Ek = 10**(0.5)       #发送能量

std = 0.01
lr = 0.00001    #学习速率0.00001
theta = 0.00002 #正则化系数

D_Net = np.array([2*K_len, 4*K_len, 2*K_len])
epoch = 1000
train_batch =30
m = train_batch
train_iter = int(epoch * m/train_batch + 1)

def batch_norm(x):
    y = copy.copy(x)
    mean = tf.reduce_mean(y)
    y = (y - mean) / tf.sqrt(tf.reduce_mean(tf.square(y - mean)))
    return y

def DNN(input):
    w_1 = tf.Variable(tf.random_normal([D_Net[0],D_Net[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,D_Net[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(md.batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([D_Net[1],D_Net[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,D_Net[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2, w_1, b_1, w_2, b_2]


Hy_date = tf.placeholder(tf.float32, shape=[None, 2 * K_len])  # [?,2k]
HH_date = tf.placeholder(tf.float32, shape=[None, 2 * K_len, 2 * K_len])  # [?,2k,2k]
target_x_date = tf.placeholder(tf.float32, shape=[None, 2 * K_len])  # [?,2k]
derta = tf.Variable(tf.ones(2,1))
X_0 = tf.zeros((train_batch, 2*K_len))    # [?,2k]   %%%%%%%%%%%%%

temp1 = tf.matmul(tf.expand_dims(X_0, 1), HH_date)  # [?,1,2k]
temp1 = tf.squeeze(temp1, 1)
X_1 = X_0 - derta[0] * Hy_date + derta[1] * temp1  # [?,2k]
[X_out,w_11,b_11,w_12,b_12] = DNN(X_1)  # [?,2k]




saver = tf.train.Saver()
saver.restore(sess, file_name)

df_w_11 = pd.DataFrame(sess.run(w_11))
df_b_11 = pd.DataFrame(sess.run(b_11))
df_w_12 = pd.DataFrame(sess.run(w_12))
df_b_12 = pd.DataFrame(sess.run(b_12))
df_derta_1 = pd.DataFrame(sess.run(derta))

df_w_11.to_csv('w_11.csv')
df_b_11.to_csv('b_11.csv')
df_w_12.to_csv('w_12.csv')
df_b_12.to_csv('b_12.csv')
df_derta_1.to_csv('derta_1.csv')
