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
            print("_____y_polit_data_____", np.shape(y_polit_data))
            # 估计H
            y_polit = y_polit_data[0:Pilot_M]   # [M, R]
            print("_____y_polit_____", np.shape(y_polit))
            y_date = y_polit_data[Pilot_M:Pilot_M+Date_N+1]  # [N,R]
            print("_____y_date_____", np.shape(y_date))
            # H_LS = sqrt(T_ChannelDim / Ex) * pinv(polit'*polit)*polit' * y_polit;
            m0 = np.transpose(np.conj(polit))
            print("_____m0_____", np.shape(m0))
            m1 = np.linalg.pinv(np.dot(m0, polit))
            print("_____m1_____", np.shape(m1))
            m2 = m1*np.transpose(np.conj(polit))
            print("_____m2_____", np.shape(m2))
            H_LS = (T_ChannelDim/Ex)**0.5 * np.dot(m2, y_polit)   # [T, R]
            # H_MMSE = sqrt(T_ChannelDim / Ex) * pinv(T_ChannelDim / Ex * eye(T_ChannelDim, T_ChannelDim) + polit'*polit)*polit' * y_polit

            # 检测data
            y_data_detect = (T_ChannelDim / Ex) ** 0.5 * np.dot(y_date, np.transpose(np.conj(H_LS)))    # [N,T]   T=1

            # 转换为实数
            x_data_re = np.concatenate((np.real(x_data), np.imag(x_data)), axis=0)   # [2N,1]
            print("_____x_data_re_____", np.shape(x_data_re))
            y_data_detect_re = np.concatenate((np.real(y_data_detect), np.imag(y_data_detect)), axis=0)    # [2N,1]
            print("_____y_data_detect_re_____", np.shape(y_data_detect_re))

            x_date.append(x_data_re)   # [2N*B,1]
            data_detect.append(y_data_detect_re)  # [2N*B,1]

        output_x_date = np.reshape(x_date, [train_batch, 2*Date_N])   #[？，2N]
        # print("===========================output_x_date========================================", output_x_date,np.shape(output_x_date))
        output_data_detect = np.reshape(data_detect, [train_batch, 2*Date_N])   # [？，2N]
        return output_x_date, output_data_detect


gen_date_train()