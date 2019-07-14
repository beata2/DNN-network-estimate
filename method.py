import tensorflow as tf
import numpy as np
import pandas as pd
import copy
sess = tf.Session()

file_name = "./t/Unit2-model-6000"
arfa = 1.
CSI_len = 16   #CSI长度
L_len = 512   #上行数据长度
Ek = 10**(0.5)       #发送能量
rou = 0.2     #叠加因子
batch = 200   #批次数
std = 0.01
lr = 0.00001    #学习速率0.00001
lr_2 = 0.000001   #学习速率0.000001
theta = 0.00002 #正则化系数

H_Net_1 = np.array([2*CSI_len, 16*CSI_len, 2*CSI_len])
D_Net_1 = np.array([2*L_len,16*L_len,2*L_len])

Gni_x_y_Net = np.array([2*L_len,16*L_len,2*L_len])



Const_h = np.float32((rou/CSI_len)**0.5)
Const_d = np.float32(((1-rou))**0.5)
Const_dd = np.float32(Const_d**(-1))

walsh = pd.read_csv('walsh_16_512.csv')  # 1024x32
walsh = walsh.astype(np.float32)
m = 100
leng = 15
epoch = 200
iteration = int(epoch*leng*m/batch + 1)


def kuo_pin(x):
    Q = walsh        #扩频向量
    out = np.dot(x,np.transpose(Q))
    return out


def batch_norm(x):
    y = copy.copy(x)
    mean = tf.reduce_mean(y)
    y = (y - mean) / tf.sqrt(tf.reduce_mean(tf.square(y - mean)))
    return y

# def batch_norm(x):
#     parm_r = tf.Variable(tf.ones([1]))
#     parm_b = tf.Variable(tf.zeros([1]))
#     y = copy.copy(x)
#     mean = tf.reduce_mean(y)
#     y = (y - mean) / (tf.sqrt(tf.reduce_mean(tf.square(y - mean))) + 0.00001)
#     return tf.add(tf.multiply(parm_r,y),parm_b)




def despreading(x):  #x向量为实数向量
    # const_temp = ((rou*Ek/CSI_len)**(-0.5)/L_len)
    y = tf.reshape(tf.stack([tf.matmul(x[:,:L_len],walsh),tf.matmul(x[:,L_len:],walsh)],axis=1),[-1,2*CSI_len])
    # return tf.multiply(const_temp,y)
    return y


def spreading(x):  #x向量为实数向量
    Real = tf.matmul(x[:,:CSI_len],np.transpose(walsh))
    Imag = tf.matmul(x[:,CSI_len:],np.transpose(walsh))
    return tf.reshape(tf.stack([Real,Imag],axis=1),[-1,2*L_len])


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
    return  num_temp/num_x


def sig_gen(M,N):
    data = (np.random.randint(0,2,[M,N])*2. - 1.) + 1j*(np.random.randint(0,2,[M,N])*2. - 1.)
    out = np.sqrt(1/2)*data
    return out

#*********************************生成训练数据集***************************************

def Noise(m):
    noise_temp = []
    for ii in range(m):

        G = ((CSI_len)**(-0.5))*(0.5**0.5)*(np.random.normal(0,1,[CSI_len,1])+1j*np.random.normal(0,1,[CSI_len,1]))
        N_mat = (Ek ** -0.5) * (0.5**0.5)*(np.random.normal(0,1,[CSI_len,L_len])+1j*np.random.normal(0,1,[CSI_len,L_len]))
        N_temp = np.dot(np.linalg.pinv(G),N_mat)
        noise_temp.append(N_temp)
    return np.reshape(noise_temp,[m,-1])
def Noise(m):
    noise_temp = []
    for ii in range(m):
        G = ((CSI_len)**(-0.5))*(0.5**0.5)*(np.random.normal(0,1,[CSI_len,1])+1j*np.random.normal(0,1,[CSI_len,1]))
        N_mat = (Ek ** -0.5) * (0.5**0.5)*(np.random.normal(0,1,[CSI_len,L_len])+1j*np.random.normal(0,1,[CSI_len,L_len]))
        N_temp = np.dot(np.linalg.pinv(G),N_mat)
        noise_temp.append(N_temp)
    return np.reshape(noise_temp,[m,-1])

def gen_data():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    for ii in range(leng):
        L_1 = sig_gen(m, L_len)
        CSI = 0.5 ** 0.5 * (np.random.normal(0, 1, [m, CSI_len]) + 1j * np.random.normal(0, 1, [m, CSI_len]))
        CSI_kuo = kuo_pin(CSI)
        T_send = (np.sqrt(1 - rou) * L_1 + np.sqrt(rou / CSI_len) * CSI_kuo)
        noise = Noise(m)
        T_data_1 = T_send + noise  # 发送数据加噪声

        T_data = np.hstack((np.real(T_data_1), np.imag(T_data_1)))  # 神经网络输入
        CSI_data = (np.hstack((np.real(CSI), np.imag(CSI))))  # CSI
        L_data = (np.hstack((np.real(L_1), np.imag(L_1))))
        input_CSI_data = T_data
        output_CSI = CSI_data  # 神经网络输出
        output_data = L_data

        x_val = pd.DataFrame(input_CSI_data)
        y_CSI = pd.DataFrame(output_CSI)
        y_data = pd.DataFrame(output_data)

        df1 = df1.append(x_val)
        df2 = df2.append(y_CSI)
        df3 = df3.append(y_data)
        print(ii)
    print(np.shape(df1))
    print(np.shape(df2))
    print(np.shape(df3))
    print(output_CSI)
    print('-' * 50)
    print(output_data)
    print('-' * 50)
    print("导出x_val")
    df1.to_csv('x_val.csv')
    print("导出y_CSI")
    df2.to_csv('y_CSI.csv')
    print("导出y_data")
    df3.to_csv('y_data.csv')



def gen_data_before():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    for ii in range(leng):
        T_data_all = []
        T_CSI = []
        T_data = []
        T_Gni_x_y = []
        for jj in range(m):
            # data数据产生
            L_1 = np.sqrt(1 / 2) * ((np.random.randint(0, 2, [1, L_len]) * 2. - 1.) + 1j * (np.random.randint(0, 2, [1, L_len]) * 2. - 1.))
            # csi产生
            CSI = 0.5 ** 0.5 * (np.random.normal(0, 1, [1, CSI_len]) + 1j * np.random.normal(0, 1, [1, CSI_len]))
            CSI_kuo = np.dot(CSI, np.transpose(walsh))

            T_send = (np.sqrt(1 - rou) * L_1 + np.sqrt(rou / CSI_len) * CSI_kuo)

            # 过信道，加噪声
            G = ((CSI_len) ** (-0.5)) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, 1]) + 1j * np.random.normal(0, 1, [CSI_len, 1]))
            noise = (Ek ** -0.5) * (0.5 ** 0.5) * (np.random.normal(0, 1, [CSI_len, L_len]) + 1j * np.random.normal(0, 1, [CSI_len, L_len]))
            Y_data = np.dot(G, T_send)+noise      # 输出的y值

            # G逆乘y
            Gni_Y_x= np.dot(np.linalg.pinv(G), Y_data)    #[1,l]

            Y_data_1 = np.reshape(Y_data, [1, -1])    #[1,c*l]
            T_data_all.append(Y_data_1)
            T_CSI.append(CSI)
            T_data.append(L_1)
            T_Gni_x_y.append(Gni_Y_x)
        T_data_all_1 = np.reshape(T_data_all, [m, -1])     #[m,c*l]
        T_CSI_1 = np.reshape(T_CSI, [m, -1])     #[m,c]
        T_data_1 = np.reshape(T_data, [m, -1])     #[m,l]
        T_Gni_x_y_1 = np.reshape(T_Gni_x_y, [m, -1])   #[m,l]


        # 神经网络输入
        T_data_all_re = np.hstack((np.real(T_data_all_1), np.imag(T_data_all_1)))    #横向拼接
        print("===========================T_data_all_re========================================",T_data_all_re.shape)  #(100, 16384)
        CSI_re = (np.hstack((np.real(T_CSI_1), np.imag(T_CSI_1))))
        print("===========================CSI_re========================================", CSI_re.shape) #(100, 32)
        L_data_re = (np.hstack((np.real(T_data_1), np.imag(T_data_1))))
        print("===========================L_data_re========================================", L_data_re.shape) #(100, 1024)
        Gni_x_y_re = (np.hstack((np.real(T_Gni_x_y_1), np.imag(T_Gni_x_y_1))))
        print("===========================Gni_x_y_re========================================", Gni_x_y_re.shape) #(100, 1024)
        input_CSI_data = T_data_all_re
        output_CSI = CSI_re  # 神经网hstack络输出
        output_data = L_data_re
        output_Gni_x_y = Gni_x_y_re

        x_val = pd.DataFrame(input_CSI_data)
        y_CSI = pd.DataFrame(output_CSI)
        y_data = pd.DataFrame(output_data)
        y_Gni_x_y = pd.DataFrame(output_Gni_x_y)

        df1 = df1.append(x_val)
        df2 = df2.append(y_CSI)
        df3 = df3.append(y_data)
        df4 = df4.append(y_Gni_x_y)
        print(ii)
    print(np.shape(df1))
    print(np.shape(df2))
    print(np.shape(df3))
    print(np.shape(df4))
    print(output_CSI)
    print('-' * 50)
    print(output_data)
    print('-' * 50)
    print(output_Gni_x_y)
    print('-' * 50)
    print("导出x_val")
    df1.to_csv('x_val.csv')
    print("导出y_CSI")
    df2.to_csv('y_CSI.csv')
    print("导出y_data")
    df3.to_csv('y_data.csv')
    print("导出output_Gni_x_y")
    df3.to_csv('output_Gni_x_y.csv')

# *********************************训练模型*****************************
def Gni_x_y_DNN(input):
    w_1 = tf.Variable(tf.random_normal([Gni_x_y_Net[0],Gni_x_y_Net[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,Gni_x_y_Net[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([Gni_x_y_Net[1],Gni_x_y_Net[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,Gni_x_y_Net[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return layer_2




# def H_DNN_1(input):
#     w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def H_DNN_1(input):
#     w_1 = np.float32(pd.read_csv('wh_11.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bh_11.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wh_12.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bh_12.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2

# def D_DNN_1(input):
#     w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def D_DNN_1(input):
#     w_1 = np.float32(pd.read_csv('wd_11.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bd_11.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), (arfa * w_1)), (arfa * b_1))))
#     w_2 = np.float32(pd.read_csv('wd_12.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bd_12.csv').ix[:,1:])
#     layer_2 = 0.5**0.5*tf.nn.tanh(10000*(tf.add(tf.matmul((layer_1),w_2),b_2)))
#     return layer_2


# def H_DNN_2(input):
#     w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def H_DNN_2(input):
#     w_1 = np.float32(pd.read_csv('wh_21.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bh_21.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wh_22.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bh_22.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# #
# def D_DNN_2(input):
#     w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def D_DNN_2(input):
#     w_1 = np.float32(pd.read_csv('wd_21.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bd_21.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), (arfa * w_1)), (arfa * b_1))))
#     w_2 = np.float32(pd.read_csv('wd_22.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bd_22.csv').ix[:,1:])
#     layer_2 = 0.5**0.5*tf.nn.tanh(10000*tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2

xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_Gni_x_y = tf.placeholder(tf.float32,shape=[None,2])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])

learn_Gni_x_y = Gni_x_y_DNN(xs)

# h_1 = despreading(xs)
# H_1 = (H_DNN_1(h_1))
# d_1 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_1))))
# D_1 = (D_DNN_1(d_1))
# h_2 = despreading(tf.subtract(xs, tf.multiply(Const_d, D_1)))
# H_2 = (H_DNN_2(h_2))
# d_2 = (tf.subtract(xs, tf.multiply(Const_h, spreading(H_2))))
# D_2 = (D_DNN_2(d_2))

#*************************训练*********************************************
def train():
    regularizer = tf.contrib.layers.l2_regularizer(theta)  # 正则化参数
    reg_term = tf.contrib.layers.apply_regularization(regularizer)


    loss_Gni_x_y = tf.reduce_mean(tf.square(target_Gni_x_y - learn_Gni_x_y))

    # loss_H = tf.reduce_mean(tf.square(target_H - H_1))/tf.reduce_mean(tf.square(target_H))
    # loss_D = tf.reduce_mean(tf.square(target_D - D_1))/tf.reduce_mean(tf.square(target_D))
    # loss = loss_H + reg_term
    # loss = loss_D + reg_term
    loss = loss_Gni_x_y+reg_term
    train = tf.train.AdamOptimizer(lr).minimize(loss)


    x_ = pd.read_csv('x_val.csv')
    H_ = pd.read_csv('y_CSI.csv')
    D_ = pd.read_csv('y_data.csv')
    Gni_x_y_ = pd.read_csv('output_Gni_x_y.csv')


    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    saver = tf.train.Saver(max_to_keep=4)
    for ii in range(iteration):
        rand_index = np.random.choice(len(x_), size=batch)
        x_input = x_.ix[rand_index, 1:]
        H_output = H_.ix[rand_index, 1:]
        D_output = D_.ix[rand_index, 1:]
        feed_dict = {xs: x_input, target_H: H_output, target_D: D_output}
        # feed_dict = {xs: x_input, target_H: H_output}
        sess.run(train, feed_dict=feed_dict)


        if ii % 1000 == 0:
            # out_data = data_yingshe(sess.run(D_1, feed_dict=feed_dict))
            # ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))

            print('-' * 50)
            print('训练%d后，loss_H = %8f.' % (ii, sess.run(loss_H, feed_dict=feed_dict)))

            # print('训练%d后，loss_D = %8f.' % (ii, sess.run(loss_D, feed_dict=feed_dict)))
            # print('训练%d后，BER = %12f.' % (ii,ber))

            # print('训练%d后，总误差Loss = %8f.' %
            #  (ii, sess.run(loss, feed_dict=feed_dict)))
            # df1 = df1.append(pd.DataFrame([(sess.run(loss_H_2,feed_dict=feed_dict)),ber]))
            # saver.save(sess, 't/Unit2-model', global_step=ii)
            saver.save(sess, 'd/Unit2-model', global_step=ii)
            # df1.to_csv('loss_MSE_BER.csv')

#******************************再训练***************************************
def train_again():
    regularizer = tf.contrib.layers.l2_regularizer(theta)  # 正则化参数
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    loss_H = tf.reduce_mean(tf.square(target_H - H_2))/tf.reduce_mean(tf.square(target_H))
    loss_D = tf.reduce_mean(tf.square(target_D - D_2))/tf.reduce_mean(tf.square(target_D))
    # loss = loss_H + reg_term
    loss = loss_D + reg_term
    train = tf.train.AdamOptimizer(lr_2).minimize(loss)

    x_ = pd.read_csv('x_val.csv')
    H_ = pd.read_csv('y_CSI.csv')
    D_ = pd.read_csv('y_data.csv')


    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,file_name)
        saver = tf.train.Saver(max_to_keep=4)
        for ii in range(iteration):
            rand_index = np.random.choice(len(x_), size=batch)
            x_input = x_.ix[rand_index, 1:]
            H_output = H_.ix[rand_index, 1:]
            D_output = D_.ix[rand_index, 1:]
            feed_dict = {xs: x_input, target_H: H_output, target_D: D_output}
            sess.run(train, feed_dict=feed_dict)
            if ii % 1000 == 0:
                out_data = data_yingshe(sess.run(D_2, feed_dict=feed_dict))
                ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))
                print('-' * 50)
                print('训练%d后，loss_H = %8f.' % (ii, sess.run(loss_H, feed_dict=feed_dict)))
                print('训练%d后，loss_D = %8f.' % (ii, sess.run(loss_D, feed_dict=feed_dict)))
                print('训练%d后，BER = %12f.' % (ii, ber))
                print('训练%d后，总误差Loss = %8f.' % (ii, sess.run(loss, feed_dict=feed_dict)))
                saver.save(sess, 't/Unit2-model', global_step=ii)

#******************************测试**************************************************
def test_model():
    L_1 = sig_gen(m,L_len)
    CSI = 0.5**0.5*(np.random.normal(0,1,[m,CSI_len])+1j*np.random.normal(0,1,[m,CSI_len]))
    CSI_kuo = kuo_pin(CSI)
    T_send = (np.sqrt(1-rou)*L_1 + np.sqrt(rou/CSI_len)*CSI_kuo)
    noise = Noise(m)
    T_data_1 = T_send + noise              #发送数据加噪声
    T_data = np.hstack((np.real(T_data_1),np.imag(T_data_1)))   #神经网络输入
    CSI_data = (np.hstack((np.real(CSI),np.imag(CSI))))  #CSI
    L_data = (np.hstack((np.real(L_1),np.imag(L_1))))
    input_CSI_data = T_data
    output_CSI = CSI_data   #神经网络输出
    output_data = L_data


    loss_H = tf.reduce_mean(tf.square(target_H - H_2))/tf.reduce_mean(tf.square(target_H))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        feed_dict = {xs: input_CSI_data, target_H: output_CSI, target_D: output_data}
        saver.restore(sess, file_name)

        out_data = data_yingshe(sess.run(D_2, feed_dict=feed_dict))
        ber = BER(out_data, data_yingshe(sess.run(target_D, feed_dict=feed_dict)))

        print('-' * 50)
        # print("期望CSI：")
        # print(output_CSI)
        # print('-' * 50)
        # print("实际CSI：")
        # print(sess.run(H_2, feed_dict=feed_dict))
        # print('-' * 50)
        print("MSE-CSI：")
        print(sess.run(loss_H, feed_dict=feed_dict))
        # print('*' * 50)

        # print('-' * 50)
        # print("期望Data：")
        # print(output_data)
        # print('-' * 50)
        # print("实际Data：")
        # print(sess.run(D_1, feed_dict=feed_dict))
        # print('-' * 50)

        # print("MSE-Data：")
        # print(sess.run(loss_D, feed_dict=feed_dict))
        print("BER-data：")
        print(ber)
        print('*' * 50)

        # print('-' * 50)
        # print(sess.run(W_H[0]))
        # print('-' * 50)
        # print(sess.run(arfa*W_D[0]))