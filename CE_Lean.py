'''
基于信道估计的学习网络
'''
import tensorflow as tf
import numpy as np
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 使用GPU/CPU
sess = tf.Session()
gpu_device_name = tf.test.gpu_device_name()
print("gpu_device_name:", gpu_device_name)

file_name = "./t/Unit2-model-87400"
DIR = "D:/homework/channel_estimation/python-channel_estimation-code"

Date_N = 50  # 数据帧长
Pilot_M = 30   # 前导数据长度
T_ChannelDim = 1   # 发送天线数   这里只能为1
R_ChannelDim = 4  # 接收天线数
Ex = 10 ** (0.0)       # 发送能量,0db信噪比
FrameLen = Date_N + Pilot_M   # 帧长
std = 0.01
# lr = 0.00001    # 学习速率0.00001
lr = 0.00001
theta = 0.00002    # 正则化系数

D_Net = np.array([2*Date_N, 4*Date_N, 2*Date_N])
epoch = 150000
train_batch = 300
m = train_batch

train_iter = int(epoch * m/train_batch + 1)



def data_yingshe(x):
    temp = copy.copy(x)
    shape = np.shape(temp)
    temp = np.reshape(temp, [1, -1])
    for ii in range(np.size(temp)):
        if (temp[0, ii] <= 0.0):
            temp[0, ii] = 0.0
        else:
            temp[0, ii] = 1.0
    return np.reshape(temp, shape)

def BER(x,y):
    num_x = np.size(x)
    temp = x-y
    num_temp = sum(sum(temp**2))
    return num_temp/num_x

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

def batch_norm(x):
    x = copy.copy(x)
    mean = tf.reduce_mean(x)
    y = (x - mean) / tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    return y
def data_gen():
    # 产生前导数据
    polit = np.sqrt(1 / 2) * ((np.random.randint(0, 2, [Pilot_M, 1]) * 2. - 1.) + 1j * (np.random.randint(0, 2, [Pilot_M, 1]) * 2. - 1.))

    x_data = np.sqrt(1 / 2) * ((np.random.randint(0, 2, [Date_N, 1]) * 2. - 1.) + 1j * (np.random.randint(0, 2, [Date_N, 1]) * 2. - 1.))
    # 数据帧长= polit + x_data
    x_polit_data = np.vstack((polit, x_data))
    return polit, x_polit_data, x_data

def gen_date_train():
    with tf.name_scope('date_gen'):
        x_date = []
        data_detect = []
        for jj in range(train_batch):
            # data数据产生
            polit, x_polit_data, x_data = data_gen()
            x_data1 = (Ex/T_ChannelDim)**0.5 * x_polit_data

            # 过信道，加噪声
            H_Vector = (0.5 ** 0.5) * (np.random.normal(0, 1, [T_ChannelDim, R_ChannelDim]) + 1j * np.random.normal(0, 1, [T_ChannelDim, R_ChannelDim]))
            noise = (Ex ** -0.5)*(0.5 ** 0.5) * (np.random.normal(0, 1, [FrameLen, R_ChannelDim]) + 1j * np.random.normal(0, 1, [FrameLen,R_ChannelDim]))
            y_polit_data = np.dot(x_data1, H_Vector)+noise      # 输出的y值  [M+N,R]
            # print("_____y_polit_data_____", np.shape(y_polit_data))
            # 估计H
            y_polit = y_polit_data[0:Pilot_M]   # [M, R]
            # print("_____y_polit_____", np.shape(y_polit))
            y_date = y_polit_data[Pilot_M:Pilot_M+Date_N+1]  # [N,R]
            # print("_____y_date_____", np.shape(y_date))
            # H_LS = sqrt(T_ChannelDim / Ex) * pinv(polit'*polit)*polit' * y_polit;
            m0 = np.transpose(np.conj(polit))
            # print("_____m0_____", np.shape(m0))
            m1 = np.linalg.pinv(np.dot(m0, polit))
            # print("_____m1_____", np.shape(m1))
            m2 = m1*np.transpose(np.conj(polit))
            # print("_____m2_____", np.shape(m2))
            H_LS = (T_ChannelDim/Ex)**0.5 * np.dot(m2, y_polit)   # [T, R]
            # H_MMSE = sqrt(T_ChannelDim / Ex) * pinv(T_ChannelDim / Ex * eye(T_ChannelDim, T_ChannelDim) + polit'*polit)*polit' * y_polit

            # 检测data
            y_data_detect = (T_ChannelDim / Ex) ** 0.5 * np.dot(y_date, np.transpose(np.conj(H_LS)))    # [N,T]   T=1

            # 转换为实数
            x_data_re = np.concatenate((np.real(x_data), np.imag(x_data)), axis=0)   # [2N,1]
            # print("_____x_data_re_____", np.shape(x_data_re))
            y_data_detect_re = np.concatenate((np.real(y_data_detect), np.imag(y_data_detect)), axis=0)    # [2N,1]
            # print("_____y_data_detect_re_____", np.shape(y_data_detect_re))

            x_date.append(x_data_re)   # [2N*B,1]
            data_detect.append(y_data_detect_re)  # [2N*B,1]

        output_x_date = np.reshape(x_date, [train_batch, 2*Date_N])   #[？，2N]
        # print("===========================output_x_date========================================", output_x_date,np.shape(output_x_date))
        output_data_detect = np.reshape(data_detect, [train_batch, 2*Date_N])   # [？，2N]
        return output_x_date, output_data_detect


# *********************************训练模型*****************************
def DNN(X_in):
    with tf.name_scope('net'):
        w_1 = tf.Variable(tf.random_normal([D_Net[0], D_Net[1]], mean=0.0, stddev=std))
        variable_summaries(w_1)
        b_1 = tf.Variable(tf.zeros([1, D_Net[1]]))
        variable_summaries(b_1)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_1)
        tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
        layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(X_in), w_1), b_1)))
        w_2 = tf.Variable(tf.random_normal([D_Net[1],D_Net[2]], mean=0.0, stddev=std))
        variable_summaries(w_2)
        b_2 = tf.Variable(tf.zeros([1, D_Net[2]]))
        variable_summaries(b_2)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_2)
        tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
        layer_2 = (tf.add(tf.matmul(layer_1, w_2), b_2))
        return layer_2

#*************************训练*********************************************
def train():
    regularizer = tf.contrib.layers.l2_regularizer(theta)  # 正则化参数
    reg_term = tf.contrib.layers.apply_regularization(regularizer)
    with tf.name_scope('loss'):
        loss_x_date = tf.reduce_mean(tf.square(target_x_date - data_afterNet))/tf.reduce_mean(tf.square(target_x_date))
        tf.summary.scalar('loss_x_date', loss_x_date)
    loss = loss_x_date+reg_term
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(lr).minimize(loss)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    saver = tf.train.Saver(max_to_keep=4)
    for ii in range(train_iter):
        output_x_date, output_data_detect = gen_date_train()
        feed_dict = {detect_x_date: output_data_detect, target_x_date: output_x_date}
        summary, _ = sess.run([merged, train], feed_dict=feed_dict)
        if ii % 500 == 0:
            out_data = data_yingshe(sess.run(data_afterNet, feed_dict=feed_dict))
            ber = BER(out_data, data_yingshe(sess.run(target_x_date, feed_dict=feed_dict)))
            print('-' * 50)
            print('训练%d后，loss_x_date = %8f.' % (ii, sess.run(loss_x_date, feed_dict=feed_dict)))
            print('训练%d后，BER = %12f.' % (ii, ber))
            saver.save(sess, 't/Unit2-model', global_step=ii)
        writer.add_summary(summary, ii)

def train_again():
    regularizer = tf.contrib.layers.l2_regularizer(theta)  # 正则化参数
    reg_term = tf.contrib.layers.apply_regularization(regularizer)
    with tf.name_scope('loss'):
        loss_x_date = tf.reduce_mean(tf.square(target_x_date - data_afterNet))/tf.reduce_mean(tf.square(target_x_date))
        tf.summary.scalar('loss_x_date', loss_x_date)
    loss = loss_x_date+reg_term
    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(lr).minimize(loss)
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:
        saver.restore(sess, file_name)
        for ii in range(train_iter):
            output_x_date, output_data_detect = gen_date_train()
            feed_dict = {detect_x_date: output_data_detect, target_x_date: output_x_date}
            summary, _ = sess.run([merged, train], feed_dict=feed_dict)
            if ii % 500 == 0:
                out_data = data_yingshe(sess.run(output_data_detect, feed_dict=feed_dict))
                ber = BER(out_data, data_yingshe(sess.run(target_x_date, feed_dict=feed_dict)))
                print('-' * 50)
                print('训练%d后，loss_x_date = %8f.' % (ii, sess.run(loss_x_date, feed_dict=feed_dict)))
                print('训练%d后，BER = %12f.' % (ii, ber))
                saver.save(sess, 't/Unit2-model', global_step=ii)
            writer.add_summary(summary, ii)

def test():
    loss_x_date = tf.reduce_mean(tf.square(target_x_date - data_afterNet)) / tf.reduce_mean(tf.square(target_x_date))
    tf.summary.scalar('loss_x_date', loss_x_date)

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, file_name)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        output_x_date, output_data_detect = gen_date_train()
        feed_dict = {detect_x_date: output_data_detect, target_x_date: output_x_date}
        summary, _ = sess.run([merged, loss_x_date], feed_dict=feed_dict)
        out_data = data_yingshe(sess.run(data_afterNet, feed_dict=feed_dict))
        ber = BER(out_data, data_yingshe(sess.run(target_x_date, feed_dict=feed_dict)))
        print('-' * 50)
        print('测试，loss_x_date = %8f.' % (sess.run(loss_x_date, feed_dict=feed_dict)))
        print('测试，BER = %12f.' %ber)
        writer.add_summary(summary)



with tf.name_scope('liucheng'):
    detect_x_date = tf.placeholder(tf.float32, shape=[None, 2 * Date_N])  # [?,2N]
    target_x_date = tf.placeholder(tf.float32, shape=[None, 2 * Date_N])  # [?,2N]

    for jj in range(1):
        data_afterNet = DNN(detect_x_date)  # [?,2N]

# train_again()
train()
# gen_date_train()
# test()