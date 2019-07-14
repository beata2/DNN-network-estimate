'''
第二层保存至csv
'''

# import method as md
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
sess = tf.Session()


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

# *********************************训练模型*****************************
def DNN_1(input):
    w_1 = np.float32(pd.read_csv('w_11.csv').iloc[:,1:])
    b_1 = np.float32(pd.read_csv('b_11.csv').iloc[:,1:])
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = np.float32(pd.read_csv('w_12.csv').iloc[:,1:])
    b_2 = np.float32(pd.read_csv('b_12.csv').iloc[:,1:])
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return layer_2
def DNN_2(input):
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
X_0 = tf.zeros((train_batch, 2*K_len))    # [?,2k]

x__ = tf.transpose(X_0)
temp1 = []
for ii in range(train_batch):
    HH_r = HH_date[ii, :, :]
    x_11 = tf.expand_dims(x__[:, ii], 1)
    t1 = tf.matmul(HH_r, x_11)
    temp1.append(t1)
print(tf.shape(temp1))
HHx = tf.reshape(temp1, [train_batch, -1])
X_1 = X_0 - np.float32(pd.read_csv('derta_1.csv').iloc[[0]].values[0][0]) * Hy_date + np.float32(pd.read_csv('derta_1.csv').iloc[[0]].values[0][1]) * HHx  # [?,2k]
X_out = DNN_1(X_1)  # [?,2k]

# 2
x__ = tf.transpose(X_out)
temp1 = []
for ii in range(train_batch):
    HH_r = HH_date[ii, :, :]
    # print(tf.shape(HH_r))
    x_11 = tf.expand_dims(x__[:, ii], 1)
    t1 = tf.matmul(HH_r, x_11)
    # print(tf.shape(t1))
    temp1.append(t1)
print(tf.shape(temp1))
HHx = tf.reshape(temp1, [train_batch, -1])
X_1 = X_0 - derta[0] * Hy_date + derta[1] * HHx  # [?,2k]
[X_out,w_21,b_21,w_22,b_22] = DNN_2(X_1)  # [?,2k]



saver = tf.train.Saver()
saver.restore(sess, file_name)

df_w_21 = pd.DataFrame(sess.run(w_21))
df_b_21 = pd.DataFrame(sess.run(b_21))
df_w_22 = pd.DataFrame(sess.run(w_22))
df_b_22 = pd.DataFrame(sess.run(b_22))
df_derta_2 = pd.DataFrame(sess.run(derta))

df_w_21.to_csv('w_21.csv')
df_b_21.to_csv('b_21.csv')
df_w_22.to_csv('w_22.csv')
df_b_22.to_csv('b_22.csv')
df_derta_2.to_csv('derta_2.csv')
