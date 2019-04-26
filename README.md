# AlexNet_




![qijiaqiao](https://github.com/DemonQiJQ/AlexNet_/blob/master/alexnet_stucture.png)

网络共八层，前五层为卷积层，后三层为全连接层，具体网络结构参数如下：
![qijiaqiao](https://github.com/DemonQiJQ/AlexNet_/blob/master/alexnet_parameters1.png)
![qijiaqiao](https://github.com/DemonQiJQ/AlexNet_/blob/master/alexnet_parameters2.png)



cifar-10数据集，共10类，包含训练集50000张32\*32\*3像素的图片，测试集10000张32\*32\*3图片

因此将输入图像由32*32转换为224*224
将最后一个全连接层的神经元数目转换为10
