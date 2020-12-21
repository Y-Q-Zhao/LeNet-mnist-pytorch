import numpy as np
import torch
import cv2
import os
import torchvision
torch.__version__

from decode_data import *

def binary(image):
    w,h=image.shape
    for i in range(w):
        for j in range(h):
            if image[i,j]<128:
                image[i,j]=1.0
            else:
                image[i,j]=0.0
    return image

class DataLoader():
    def __init__(self,batch_size=100):
        self.train_x,self.train_y,self.test_x,self.test_y=self.load_mnist()
        self.batch_size=batch_size
        self.batch_index=0
        self.test_batch_index=0
        self.batch_num=len(self.train_x)//self.batch_size
        self.test_batch_num=len(self.test_x)//self.batch_size

    def load_mnist(self):
        train_x,train_y,test_x,test_y=load_all()
        # L_train,=train_y.shape
        # L_test,=test_y.shape

        # train_data=[]
        # train_label=[]
        # test_data=[]
        # test_label=[]
        # for i in range(L_train):
        #     if train_y[i]==class_1 or train_y[i]==class_2:
        #         train_data.append(train_x[i,:,:])
        #         train_label.append(train_y[i])
        # for j in range(L_test):
        #     if test_y[j]==class_1 or test_y[j]==class_2:
        #         test_data.append(test_x[j,:,:])
        #         test_label.append(test_y[j])

        # print(len(train_data),len(train_label),len(test_data),len(test_label))
        return train_x,train_y,test_x,test_y

    def load_batch(self):
        batch_x=self.train_x[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size,:,:]
        batch_y=self.train_y[self.batch_index*self.batch_size:(self.batch_index+1)*self.batch_size]

        self.batch_index=(self.batch_index+1)%self.batch_num

        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        batch_=[]
        for i in range(self.batch_size):
            # batch_x[i,:,:]=binary(batch_x[i,:,:])
            batch_.append(transformer(batch_x[i,:,:]))

        batch_x = torch.stack(batch_, dim=0)

        # batch_x=np.expand_dims(batch_x,axis=1)

        batch_x=torch.tensor(batch_x,dtype=torch.float32)
        batch_y=torch.tensor(batch_y,dtype=torch.long)

        return [batch_x,batch_y]

    def load_test_batch(self):
        batch_x=self.test_x[self.test_batch_index*self.batch_size:(self.test_batch_index+1)*self.batch_size,:,:]
        batch_y=self.test_y[self.test_batch_index*self.batch_size:(self.test_batch_index+1)*self.batch_size]

        self.test_batch_index=(self.test_batch_index+1)%self.test_batch_num

        transformer = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

        batch_=[]
        for i in range(self.batch_size):
            # batch_x[i,:,:]=binary(batch_x[i,:,:])
            batch_.append(transformer(batch_x[i,:,:]))

        batch_x = torch.stack(batch_, dim=0)

        # batch_x=np.expand_dims(batch_x,axis=1)

        batch_x=torch.tensor(batch_x,dtype=torch.float32)
        batch_y=torch.tensor(batch_y,dtype=torch.long)

        return [batch_x,batch_y]



if __name__ == '__main__':
    l2c=DataLoader()
    batch=l2c.load_batch()
    batch_image=batch[0]
    image_tensor=batch_image[0,0]
    image=image_tensor.numpy()
    print(image.shape)
    (a,b)=image.shape
    for i in range(a):
        for j in range(b):
            if image[i,j]<0:
                print("pixel<0 exist")
            if image[i,j]>0:
                print("pixel>0 exist")