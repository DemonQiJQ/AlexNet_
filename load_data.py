'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
'''
import pickle
import numpy as np
import os
from skimage import io
import cv2

def resize_image(x):
    '''
    input size: n*32*32*3
    output size: n*224*224*3
    '''
    data = []
    num = len(x)
    for i in range(num):
        data.append(cv2.resize(x[i],(224,224)))
    #x = np.concatenate(data)
    x = np.array(data)
    return x

def load_cifar_batch(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
        x = dict['data']
        y = dict['labels']
        x  = x.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        y = np.array(y)
        return x,y

def load_cifar_train():
    filename_ = 'cifar-10-batches-py/data_batch_'
    data = []
    labels = []
    for i in range(5):
        filename = filename_ + str(i+1)
        data_,labels_ = load_cifar_batch(filename)
        data.append(data_)
        labels.append(labels_)

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data,labels

def slice_batch(data,slice_num):
    batch_sum = len(data)
    per_num = batch_sum/slice_num
    #new_slice = []
    slice_idx = np.arange(batch_sum)
    for i in range(slice_num):
        batch_idx = slice_idx[i * per_num : (i + 1) * per_num]
        new_slice = [data[k] for k in batch_idx]
        #new_slice.append(data[i * per_num + 1 : (i + 1) * per_num])
    #new_slice.append(data[(slice_num - 1) * per_num + 1 : batch_sum])
    #result = np.array(new_slice)
    return new_slice

def load_cifa_test():
    data,labels = load_cifar_batch('cifar-10-batches-py/test_batch')
    return data,labels

def load_pkl(slice_num):
    with open('CIFAR-10-train-label.pkl','rb') as f:
        training_data = pickle.load(f)
        training_num = len(training_data)
        small_batch_num = int(training_num / slice_num)
        for s in range(slice_num):
            batch_idx = np.arange(train_num)[s * small_batch_num : (s + 1) * small_batch_num]
            batch = [training_data[k] for k in batch_idx]
            batch_x = []
            batch_y = []
            for b in batch:
                batch_x.append(cv2.resize(cv2.imread(b[0]),(224,224))/255)
                batch_y.append(b[1])
            batch_x = np.array(batch_x,dtype=np.float32)
            batch_y = np.array(batch_y,dtype=np.float32)


                    

def output_img():
    x,y = load_cifar_batch('cifar-10-batches-py/data_batch_1')
    image_num = len(x)

    root = "images"
    if os.path.isdir(root):
        print("目录已存在")
    else:
        os.makedirs(root)
    if not os.path.isdir(root):
        print("创建目录失败")
        return

    for i in range(image_num):
        basename = os.path.join(root,"image_")
        name = basename + str(i) + ".png"
        io.imsave(name,x[i])