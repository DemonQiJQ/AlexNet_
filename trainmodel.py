import cnn_
import AlexNet
import load_data
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import pickle
import cv2

LEARNING_RATE = 1e-4
DROPOUT = 0.5
EPOCH = 10
BATCH = 5
slice_num = 200
model_path = "model"
model_name = "_cifar10.ckpt"

def get_batch(data,label,batch_size):
    input_queue = tf.train.slice_input_producer([data,label],num_epochs=1,shuffle=True,capacity=32)
    x_batch,y_batch = tf.train.batch(input_queue,batch_size=batch_size,num_threads=1,capacity=32,\
        allow_smaller_final_batch=False)
    return x_batch,y_batch

def train():
    input_data = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    input_labels = tf.placeholder(dtype=tf.float32,shape=[None,1])
    keep_prob_placeholder = tf.placeholder(dtype=tf.float32)
    
    output_alexnet = cnn_.AlexNet(input_data)
    cost,accuracy = cnn_.Alex_loss(output_alexnet,input_labels)
    #output_alexnet = AlexNet.AlexNet(input_data,10,0.5)
    #cost,accuracy = AlexNet.AlexNet_loss(output_alexnet,input_labels)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCH):
            #print("------EPOCH:",(epoch+1))
            file_basename = "cifar-10-batches-py/data_batch_"
            for batch in range(BATCH):
                #print("----------------batch:",(batch+1))
                filename = file_basename + str(batch+1)
                train_x_small,train_y_small = load_data.load_cifar_batch(filename)
                train_num = len(train_x_small)
                small_batch_num = int(train_num/slice_num)
                arrange_idx = np.arrange(train_num)
                arrange_idx_shuffle = np.random.shuffle(arrange_idx)#数据打乱
                for s in range(slice_num):                    
                    batch_idx = arrange_idx_shuffle[s * small_batch_num : (s + 1) * small_batch_num]
                    train_x_small_slice = [train_x_small[k] for k in batch_idx]
                    train_y_small_slice = [train_y_small[k] for k in batch_idx]
                    train_x = load_data.resize_image(train_x_small_slice)/255
                    train_y = np.array(train_y_small_slice)
                    
                    train_y = np.reshape(train_y,(small_batch_num,1))
                    _, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={input_data:train_x,input_labels:train_y, keep_prob_placeholder:DROPOUT})
                    print("-----------epoch:%d--------------batch:%d----------loss:%f,accuracy:%f" %(epoch+1,batch+1,loss,acc))
        model_name_ = str(EPOCH) + model_name
        saver.save(sess, os.path.join(model_path, model_name_))

    
def train_():
    input_data = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3])
    input_labels = tf.placeholder(dtype=tf.float32,shape=[None,10])
    keep_prob_placeholder = tf.placeholder(dtype=tf.float32)
    
    output_alexnet = cnn_.AlexNet(input_data)
    cost,accuracy = cnn_.Alex_loss(output_alexnet,input_labels)
    #train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with open('CIFAR-10-train-label.pkl','rb') as f:
            training_data = pickle.load(f)
            training_num = len(training_data)
            small_batch_num = int(training_num / slice_num)
            for epoch in range(EPOCH):
                print("------EPOCH:",epoch)
                for s in range(slice_num):
                    print("-------------batch:",s)
                    batch_idx = np.arange(training_num)[s * small_batch_num : (s + 1) * small_batch_num]
                    batch = [training_data[k] for k in batch_idx]
                    batch_x = []
                    batch_y = []
                    for b in batch:
                        batch_x.append(cv2.resize(cv2.imread(b[0]),(224,224))/255)
                        batch_y.append(b[1])
                    train_x = np.array(batch_x,dtype=np.float32)
                    
                    train_y = np.reshape(np.array(batch_y,dtype=np.float32),(small_batch_num,1))

                    train_y_new = np.zeros((small_batch_num,10))
                    for m in range(small_batch_num):
                        train_y_new[m][int(train_y[m])] = 1
                    '''
                    print("train_x:",np.shape(train_x))
                    print(train_x[1])
                    print("train_y:",np.shape(train_y))
                    print("train_y_new",np.shape(train_y_new))
                    '''

                    _, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={input_data:train_x,input_labels:train_y_new, keep_prob_placeholder:DROPOUT})
                    print("loss:%f,accuracy:%f" %(loss,acc))
            model_name_ = str(EPOCH) + model_name
            saver.save(sess, os.path.join(model_path, model_name_))



if __name__ == '__main__':
    #train()
    train_()