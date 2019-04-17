import numpy as np
import tensorflow as tf
import time
import math
from datetime import datetime

keep_prob = 0.5
batch_size = 32
num_batch = 100

def print_architecture(t):
    """print the architecture information of the network,include name and size"""
    print(t.op.name," ",t.get_shape().as_list())
 

def model_build(images):

    parameters = []

    with tf.name_scope('conv1') as scope:
        '''
        input_data: 227*227*3
        kernel_size: 11*11*96
        strides: 4*4
        conv1_output: 55*55*96
        pool_kernel: 3*3
        pool_strides: 2*2
        output_size: 27*27*96
        '''
        kernel = tf.Variable(tf.truncated_normal([11,11,3,96],dtype=tf.float32,stddev=0.1),name='weights')
        conv =  tf.nn.conv2d(images,kernel,strides=[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        print_architecture(conv1)
        parameters += [kernel,biases]

        #LRN
        lrn1 = tf.nn.lrn(conv1,depth_radius=4,bias=1,alpha=0.001/9,beta=0.75,name='lrn1')

        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        print_architecture(pool1)

    with tf.name_scope('conv2') as scope:
        '''
        input_data: 27*27*96
        kernel_size: 5*5*256
        strides: 1*1
        pad: 2
        conv2_output: 27*27*256
        pool_kernel: 3*3
        pool_strides: 2*2
        output_size: 13*13*256
        '''
        kernel = tf.Variable(tf.truncated_normal([5,5,96,256],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        print_architecture(conv2)
        parameters += [kernel,biases]
        lrn2 = tf.nn.lrn(conv2,depth_radius=4,bias=1,alpha=0.001/9,beta=0.75,name='lrn2')
        pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
        print_architecture(pool2)



    with tf.name_scope('conv3') as scope:
        '''
        input_data: 13*13*256
        kernel_size: 3*3*384
        strides: 1*1
        pad: 1
        conv3_output: 13*13*384
        '''
        kernel = tf.Variable(tf.truncated_normal([3,3,256,384],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(pool2,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        print_architecture(conv3)
        parameters += [kernel,biases]


    with tf.name_scope('conv4') as scope:
        '''
        input_data: 13*13*384
        kernel_size: 3*3*384
        strides: 1*1
        pad: 1
        conv4_output: 13*13*384
        '''
        kernel = tf.Variable(tf.truncated_normal([3,3,384,384],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        print_architecture(conv4)
        parameters += [kernel,biases]

    with tf.name_scope('conv5') as scope:
        '''
        input_data: 13*13*384
        kernel_size: 3*3*256
        strides: 1*1
        pad: 1
        conv3_output: 13*13*256

        pool_kernel: 3*3
        pool_strides: 2*2
        pool_output: 6*6*256
        '''
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=0.1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name=scope)
        print_architecture(conv5)
        parameters += [kernel,biases]

        pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
        print_architecture(pool5)

    with tf.name_scope('fc6') as scope:
        """
        input:pool5 [6*6*256]
        output:fc6 [4096]
        """
        kernel = tf.Variable(tf.truncated_normal([6*6*256,4096], dtype=tf.float32, stddev=0.1), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),trainable=True, name="biases")
        # 输入数据变换
        flat = tf.reshape(pool5, [-1, 6*6*256] )  # 整形成m*n,列n为7*7*64
        # 进行全连接操作
        fc = tf.nn.relu(tf.matmul(flat, kernel) + biases,name='fc6')
        # 防止过拟合  nn.dropout
        fc6 = tf.nn.dropout(fc, keep_prob)
        parameters += [kernel, biases]
        print_architecture(fc6)

    with tf.name_scope('fc7') as scope:
        """
        input:pool5 [4096]
        output:fc7 [4096]
        """
        kernel = tf.Variable(tf.truncated_normal([4096,4096], dtype=tf.float32, stddev=0.1), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),trainable=True, name="biases")
        # 进行全连接操作
        fc = tf.nn.relu(tf.matmul(fc6, kernel) + biases,name='fc7')
        # 防止过拟合  nn.dropout
        fc7 = tf.nn.dropout(fc, keep_prob)
        parameters += [kernel, biases]
        print_architecture(fc7)

    with tf.name_scope('fc8') as scope:
        """
        input:pool5 [4096]
        output:fc8 [1000]
        """
        kernel = tf.Variable(tf.truncated_normal([4096,1000], dtype=tf.float32, stddev=0.1), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32),trainable=True, name="biases")
        # 进行全连接操作
        fc = tf.nn.relu(tf.matmul(fc7, kernel) + biases,name='fc8')
        # 防止过拟合  nn.dropout
        fc8 = tf.nn.dropout(fc, keep_prob)
        parameters += [kernel, biases]
        print_architecture(fc8)

    return fc8,parameters

 
def time_compute(session,target,info_string):
    num_step_burn_in=10  #预热轮数，头几轮迭代有显存加载、cache命中等问题可以因此跳过
    total_duration=0.0   #总时间
    total_duration_squared=0.0
    for i in range(num_batch+num_step_burn_in):
        start_time=time.time()
        _ = session.run(target)
        duration= time.time() -start_time
        if i>= num_step_burn_in:
            if i%10==0: #每迭代10次显示一次duration
                print("%s: step %d,duration=%.5f "% (datetime.now(),i-num_step_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration *duration
    time_mean=total_duration /num_batch
    time_variance=total_duration_squared / num_batch - time_mean*time_mean
    time_stddev=math.sqrt(time_variance)
    #迭代完成，输出
    print("%s: %s across %d steps,%.3f +/- %.3f sec per batch "%
              (datetime.now(),info_string,num_batch,time_mean,time_stddev))




def main():
    with tf.Graph().as_default():
        """仅使用随机图片数据 测试前馈和反馈计算的耗时"""
        image_size =224
        images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=0.1 ) )
        fc8,parameters=model_build(images)
 
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)

    
        time_compute(sess,target=fc8,info_string="Forward")
 
        obj=tf.nn.l2_loss(fc8)
        grad=tf.gradients(obj,parameters)
        time_compute(sess,grad,"Forward-backward")



if __name__=='__main__':
    main()
