#coding:utf-8
__author__ = 'dk'
#特征提取网络
import  torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
class Deep_fingerprinting(nn.Module):
    def __init__(self,feature_width=100,nb_classes=None):
        #初始化
        super(Deep_fingerprinting,self).__init__()
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]
        #block1
        self._1Conv1D = nn.Conv1d(stride=conv_stride_size[1],
                                 kernel_size=kernel_size[1],
                                 in_channels=1,
                                 out_channels=filter_num[1],
                                 padding=kernel_size[1]//2)
        self._2BatchNormalization = nn.BatchNorm1d(filter_num[1])
        self._3ELU = nn.ELU(alpha=1.0)
        self._4Conv1D = nn.Conv1d(in_channels=filter_num[1],
                                  out_channels=filter_num[1],
                                  kernel_size = kernel_size[1],
                                  stride=conv_stride_size[1],
                                  padding=kernel_size[1]//2)
        self._5BatchNormalization = nn.BatchNorm1d(filter_num[1])
        self._6ELU = nn.ELU(alpha=1.0)
        self._7MaxPooling1D=nn.MaxPool1d(stride=pool_stride_size[1],
                                         kernel_size=pool_size[1],padding=1)
        self._8Dropout = nn.Dropout(p=0.1)

        #block2
        self._9Conv1D = nn.Conv1d(in_channels=filter_num[1],
                                  out_channels=filter_num[2],
                                  kernel_size = kernel_size[2],
                                  stride=conv_stride_size[2],
                                  padding=kernel_size[2]//2)
        self._10BatchNormalization = nn.BatchNorm1d(filter_num[2])
        self._11Relu = nn.ReLU()
        self._12Conv1D = nn.Conv1d(in_channels=filter_num[2],
                                  out_channels=filter_num[2],
                                  kernel_size = kernel_size[2],
                                  stride=conv_stride_size[2],
                                  padding=kernel_size[2]//2)
        self._13BatchNormalization = nn.BatchNorm1d(filter_num[2])
        self._14Relu = nn.ReLU()
        self._15MaxPooling1D=nn.MaxPool1d(kernel_size=pool_size[2],
                                          stride=pool_stride_size[2],
                                          padding=pool_size[2]//2)
        self._16Dropout = nn.Dropout(p=0.1)

        #block3
        self._17Conv1D = nn.Conv1d(in_channels=filter_num[2],
                                  out_channels=filter_num[3],
                                  kernel_size = kernel_size[3],
                                  stride=conv_stride_size[3],
                                  padding=kernel_size[3]//2)
        self._18BatchNormalization = nn.BatchNorm1d(filter_num[3])
        self._19Relu = nn.ReLU()
        self._20Conv1D = nn.Conv1d(in_channels=filter_num[3],
                                  out_channels=filter_num[3],
                                  kernel_size = kernel_size[3],
                                  stride=conv_stride_size[3],
                                  padding=kernel_size[3]//2)
        self._21BatchNormalization = nn.BatchNorm1d(filter_num[3])
        self._22Relu = nn.ReLU()
        self._23MaxPooling1D=nn.MaxPool1d(kernel_size=pool_size[3],
                                          stride=pool_stride_size[3],
                                          padding=pool_size[3]//2)
        self._24Dropout = nn.Dropout(p=0.1)

        #Block4
        self._25Conv1D = nn.Conv1d(in_channels=filter_num[3],
                                  out_channels=filter_num[4],
                                  kernel_size = kernel_size[4],
                                  stride=conv_stride_size[4],
                                  padding=kernel_size[4]//4)
        self._26BatchNormalization = nn.BatchNorm1d(filter_num[4])
        self._27Relu = nn.ReLU()
        self._28Conv1D = nn.Conv1d(in_channels=filter_num[4],
                                  out_channels=filter_num[4],
                                  kernel_size = kernel_size[4],
                                  stride=conv_stride_size[4],
                                  padding=kernel_size[4]//2)
        self._29BatchNormalization = nn.BatchNorm1d(filter_num[4])
        self._30Relu = nn.ReLU()
        self._31MaxPooling1D=nn.MaxPool1d(kernel_size=pool_size[4],
                                          stride=pool_stride_size[4],
                                          padding=pool_size[4]//2)
        self._32Dropout = nn.Dropout(p=0.1)

        self._33Flattern = nn.Flatten()
        self._34Dense = nn.Sequential(nn.Linear(in_features=1024,out_features=feature_width),nn.ReLU(True))
        if nb_classes!= None:
            self._35Sigmoid = nn.Sequential(nn.Linear(in_features=feature_width,out_features=nb_classes),nn.Sigmoid())
        else:
            self._35Sigmoid = None
    def forward(self, x):
        x = self._1Conv1D(x)
        x = self._2BatchNormalization(x)
        x = self._3ELU(x)
        x = self._4Conv1D(x)
        x = self._5BatchNormalization(x)
        x = self._6ELU(x)
        x = self._7MaxPooling1D(x)
        x = self._8Dropout(x)

        #block2
        x = self._9Conv1D(x)
        x = self._10BatchNormalization(x)
        x = self._11Relu(x)
        x = self._12Conv1D(x)
        x = self._13BatchNormalization(x)
        x = self._14Relu(x)
        x = self._15MaxPooling1D(x)
        x = self._16Dropout(x)

        #block3
        x = self._17Conv1D(x)
        x = self._18BatchNormalization(x)
        x = self._19Relu(x)
        x = self._20Conv1D(x)
        x = self._21BatchNormalization(x)
        x = self._22Relu(x)
        x = self._23MaxPooling1D(x)
        x = self._24Dropout(x)

        #block4
        x = self._25Conv1D(x)
        x = self._26BatchNormalization(x)
        x = self._27Relu(x)
        x = self._28Conv1D(x)
        x = self._29BatchNormalization(x)
        x = self._30Relu(x)
        x = self._31MaxPooling1D(x)
        x = self._32Dropout(x)

        #feature extrator
        x = self._33Flattern(x)
        x = self._34Dense(x)

        #sigmoid
        if self._35Sigmoid!=None:
            x = self._35Sigmoid(x)
        return  x

def test_df():
    df = Deep_fingerprinting(35)
    import numpy as np
    from keras.utils import to_categorical
    x = th.rand((3,1,1000),dtype=th.float)
    y = to_categorical([0,1,2],35)
    summary(df,input_size=(1,1000))
    predict_y = df(x)
    print(predict_y)
    df.zero_grad()

class AWF(nn.Module):
    def __init__(self,feature_width=100,nb_classes=None):
        #初始化
        super(AWF,self).__init__()

        #build the layers

        #The first layer
        self._1Conv1D=nn.Conv1d(in_channels=1,
                                out_channels=32,
                                kernel_size=5,
                                stride=1)
        self._2Relu=nn.ReLU()
        #The second layers
        self._3Conv1D=nn.Conv1d(in_channels=32,
                                out_channels=32,
                                kernel_size=5,
                                stride=1)
        self._4Relu=nn.ReLU()

        #The third layers
        self._5Maxpooling=nn.MaxPool1d(kernel_size=4)

        #The 4th layers
        self._6Flattern=nn.Flatten(start_dim=1)

        self._7LSTM=nn.LSTM(input_size=256,
                            hidden_size=feature_width*2)
        #The last layers
        self._8Dense = nn.Sequential(nn.Linear(in_features=feature_width*2,out_features=feature_width),nn.ReLU(True))
        if nb_classes!=None:
            self._8Sigmoid = nn.Sequential(nn.Linear(in_features=feature_width,out_features=nb_classes),nn.Sigmoid())
        else:
            self._8Sigmoid = None
    def forward(self, x):
        x = self._1Conv1D(x)
        x = self._2Relu(x)
        x = self._3Conv1D(x)
        x = self._4Relu(x)
        x = self._5Maxpooling(x)

        ##feature extract
        x = self._6Flattern(x)
        x = th.reshape(x,(31,-1,256))
        print(x.shape)
        x,(_,__) = self._7LSTM(x)
        x = self._8Dense(x[-1])
        print(x.shape)
        if self._8Sigmoid!= None:
            x = self._8Sigmoid(x)
        print(x.shape)
        return  x
class IDENTIFY(nn.Module):
    def __init__(self,feature_width=100,nb_classes=None):
        super(IDENTIFY,self).__init__()
        #self.Dense = nn.Sequential(nn.Linear(in_features=feature_width*2,out_features=feature_width),nn.ReLU(True))
    def forward(self, x):
        return x

def test_awf():
    awf = AWF(40,nb_classes=3)
    import numpy as np
    from keras.utils import to_categorical
    x = th.rand((3,1,1000),dtype=th.float)
    y = to_categorical([0,1,2],3)
    #summary(awf,input_size=(1,1000))
    predict_y = awf(x)
    print(predict_y)
    awf.zero_grad()
if __name__ == '__main__':
    test_df()
