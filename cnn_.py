import tensorflow as tf

keep_prob = 0.5

def conv_layer(layer_name,input_data,channels,kernel_size,kernel_num,strides,Lrn=0):
        with tf.name_scope(layer_name) as scope:
                kernel = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,channels,kernel_num],dtype=tf.float32,stddev=0.1),name='weights')
                conv = tf.nn.conv2d(input_data,kernel,[1,strides,strides,1],padding='SAME')
                biases = tf.Variable(tf.constant(0.0,shape=[kernel_num],dtype=tf.float32),trainable=True,name='biases')
                bias = tf.nn.bias_add(conv,biases)
                conv_ = tf.nn.relu(bias,name=scope)
                if Lrn == 1:
                        lrn_ = tf.nn.lrn(conv_,depth_radius=4,bias=1,alpha=0.001/9,beta=0.75)
                        pool = tf.nn.max_pool(lrn_,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                        return pool
        return conv_


def AlexNet(data):
        '''
        input_data: 224*224*3
        '''
        conv1 = conv_layer('conv1',data,channels=3,kernel_size=11,kernel_num=96,strides=4,Lrn=1)
        conv2 = conv_layer('conv2',conv1,channels=96,kernel_size=5,kernel_num=256,strides=1,Lrn=1)
        conv3 = conv_layer('conv3',conv2,channels=256,kernel_size=3,kernel_num=384,strides=1,Lrn=0)
        conv4 = conv_layer('conv4',conv3,channels=384,kernel_size=3,kernel_num=384,strides=1,Lrn=0)
        conv5 = conv_layer('conv5',conv4,channels=384,kernel_size=3,kernel_num=256,strides=1,Lrn=1)
        conv5_ = tf.reshape(conv5,[-1,6*6*256])
        
        fc6 = tf.layers.dense(conv5_,1024,activation=tf.nn.relu,name="fc6")
        fc6_ = tf.nn.dropout(fc6,keep_prob)
        fc7 = tf.layers.dense(fc6_,128,activation=tf.nn.relu,name="fc7")
        fc7_ = tf.nn.dropout(fc7,keep_prob)
        fc8 = tf.layers.dense(fc7_,10,activation=None,name="fc8")
        
        return fc8

def Alex_loss(y,y_hat):
        y_hat = tf.cast(y_hat,dtype=tf.float32)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_hat))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),tf.argmax(y_hat,1)), tf.float32))
        return loss, accuracy





