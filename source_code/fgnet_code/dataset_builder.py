__author__ = 'dk'
### APP网络流行为图数据集建立
import random
import time
import gzip
import os
import sys
import dgl
import pickle
import logger_wrappers
import torch as th
import  numpy as np
from graph_neural_network_model_v2.construct_graph import build_graphs_from_json,mtu,mtime, concurrent_time_threshold
import tqdm

root_dir  =  "./"
random.seed(100)
time_period= 60

class Dataset:
    def __init__(self,graph_json_directory=root_dir+"Traffic_graph_generator/graph/",mode='clear',dumpData=False,usedumpData=False,dumpFilename="dataset_builder.pkl.gzip",cross_version=False,test_split_rate=0.1):
        '''
        :param graph_json_directory:  图json数据目标
        :param mode:                  使用clear数据(`clear`)还是带噪声的noise数据(`noise`),还是两种数据集都兼顾(`all`)
        :param dumpData:              每次导入数据都特别慢,当dumpData打开的时候,会把数据直接提取出来
        :param test_split_rate:       数据里面划分出来的测试集占的总样本的比例,默认10%.其中一半拿来做test dataset，一半拿来做validate dataset
        :param dumpFilename:          把数据导出到那个缓存文件,或者从那个缓存文件导入数据
        :param cross_version:         True: 跨版本的数据集
        :return:
        '''
        self.dumpFileName = dumpFilename
        if usedumpData==True and os.path.exists(dumpFilename):
            fp = gzip.GzipFile(dumpFilename,"rb")
            data=pickle.load(fp)
            fp.close()
            self.labelName = data['labelName']
            self.labelNameSet=data['labelNameSet']
            self.graphs = data['graphs']
            self.labelId = data['labelId']
            self.train_index = data['train_index']
            self.test_index = data['test_index']
            self.valid_index = data['valid_index']
            info ='Load dump data from {0}'.format(dumpFilename)
            logger_wrappers.warning(info)
        else:
            if os.path.isdir(graph_json_directory)== False:
                info = '{0} is not a directory'.format(graph_json_directory)
                logger_wrappers.error(info)
                raise BaseException(info)
            assert mode in ['clear','noise','all']
            self.labelName = []
            self.labelNameSet={}
            self.labelId  =[]
            self.graphs =[]
            #self.flowCounter = 0

            _labelNameSet =[]
            for _root,_dirs,_files in os.walk(graph_json_directory):
                if _root == graph_json_directory or len(_files)==0:
                    continue
                _root =_root.replace("\\","/")
                versionName = _root.split("/")[-1]
                packageName = _root.split("/")[-1]       #app数据
                #packageName = _root.split('/')[-1]      #区块链数据
                labelName=packageName
                _labelNameSet.append(labelName)
            _labelNameSet.sort()

            for i in range(len(_labelNameSet)):
                self.labelNameSet.setdefault(_labelNameSet[i], len(self.labelNameSet))

            for _root,_dirs,_files in os.walk(graph_json_directory):
                if _root == graph_json_directory or len(_files)==0:
                    continue
                _root =_root.replace("\\","/")
                versionName = _root.split("/")[-1]
                packageName = _root.split("/")[-1]     ## app做法
                #packageName = _root.split('/')[-1]      ## 区块链
                labelName=packageName
                print(labelName)
                #random.shuffle(_files)
                for index in tqdm.trange(len(_files)):
                    file = _files[index]
                    json_fname = (_root +"\\" + file).replace("\\","/")

                    ##以下是给app pcap进行过滤的
                    #if mode != 'all' and mode not in file:
                    #    continue

                    gs = build_graphs_from_json(json_fname, time_period=time_period)
                    if len(gs) < 1 or gs[0] == None:
                        continue
                    assert len(gs) == 1
                    self.graphs+= gs
                    self.labelName+= [labelName] * len(gs)
                    self.labelId+= [self.labelNameSet[labelName]] * len(gs)
                    assert self.labelId[-1] in range(len(self.labelNameSet))

            assert len(self.graphs) == len(self.labelId)
            assert len(self.graphs) == len(self.labelName)

            info = "Build {0} graph over {1} classes, {2} graph per class. {3} flow.".format(len(self.graphs),len(self.labelNameSet),len(self.graphs)//len(self.labelNameSet),self.flowCounter)
            logger_wrappers.info(info)
            print(info)
            ###划分训练集,验证集,测试集
            ###比例: 90: 05 : 05
            self.train_index = []
            self.valid_index = []
            self.test_index =  []

            #随机打乱
            index =list( range(len(self.graphs)))
            random.shuffle(index)
            test_split = int(test_split_rate * 100)//2
            for i in index:
                r = random.randint(0,100)
                if r in range(0,test_split):
                    self.test_index.append(i)
                elif r in range(test_split,2*test_split):
                    self.valid_index.append(i)
                else:
                    self.train_index.append(i)

            if dumpData :
                self.dumpData()
        self.class_aliasname ={}        #label的可读别名
        labelNameSet = list(self.labelNameSet)
        labelNameSet.sort()             #对标签排序
        for i in range(len(labelNameSet)):
            self.class_aliasname.setdefault(i,labelNameSet[i])
        print('Train:{0},Test:{1},Valid:{2}'.format(len(self.train_index),len(self.test_index),len(self.valid_index)))

        self.train_watch = 0
        self.test_watch =  0
        self.valid_watch = 0
        self.epoch_over = False

    def dumpData(self,dumpFileName=None):

        if dumpFileName == None:

            dumpFileName = self.dumpFileName

        fp= gzip.GzipFile(dumpFileName,"wb")
        pickle.dump({
                'graphs':self.graphs,
                'flowCounter':self.flowCounter,
                'labelName':self.labelName,
                'labelNameSet':self.labelNameSet,
                'labelId':self.labelId,
                'train_index':self.train_index,
                'valid_index':self.valid_index,
                'test_index':self.test_index
            },file=fp,protocol=-1)
        fp.close()

    def __next_batch(self,name,batch_size):
        graphs =[]
        labels =[]

        for i in range(batch_size):
            if name == 'train':
                graphs.append(self.graphs[self.train_index[self.train_watch]])
                labels.append(self.labelId[self.train_index[self.train_watch]])

                if (self.train_watch + 1) == len(self.train_index):
                    self.epoch_over +=1

                self.train_watch = (self.train_watch + 1) % len(self.train_index)
            elif name =='valid':
                graphs.append(self.graphs[self.valid_index[self.valid_watch]])
                labels.append(self.labelId[self.valid_index[self.valid_watch]])
                self.valid_watch = (self.valid_watch + 1) % len(self.valid_index)
            else:
                graphs.append(self.graphs[self.test_index[self.test_watch]])
                labels.append(self.labelId[self.test_index[self.test_watch]])
                self.test_watch = (self.test_watch + 1) % len(self.test_index)
        return dgl.batch(graphs),th.tensor(labels)
    def next_train_batch(self,batch_size):
        return self.__next_batch('train',batch_size)
    def next_valid_batch(self,batch_size):
        return self.__next_batch('valid',batch_size)
    def next_test_batch(self,batch_size):

        return self.__next_batch('test',batch_size)
    def export_wf_dataset(self,path_dir,feature_name='pkt_length'):
        #把数据导出成wf-attacks模型可以处理的数据形式
        if os.path.exists(path_dir)== False:
            os.makedirs(path_dir)
        assert  feature_name in ['pkt_length','arv_time']
        X_train =[]
        y_train =[]
        X_valid =[]
        y_valid =[]
        X_test =[]
        y_test =[]
        #print(self.train_index)
        #print(self.test_index)
        #print(self.valid_index)
        #print(self.labelId)
        #print('graph totoal:',len(self.graphs))
        for i in self.train_index:
            X_train.append(self.graphs[i].ndata[feature_name]*mtu)
            y_train += [self.labelId[i]] * len(self.graphs[i].nodes())
        for i in self.test_index:
            X_test.append(self.graphs[i].ndata[feature_name]*mtu)
            y_test += [self.labelId[i]] * len(self.graphs[i].nodes())
        for i in self.valid_index:
            X_valid.append(self.graphs[i].ndata[feature_name]*mtu)
            y_valid += [self.labelId[i]] * len(self.graphs[i].nodes())

        #合并X
        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        X_valid =np.concatenate(X_valid)

        X_train = np.reshape(X_train,(-1,1000,1))
        X_test = np.reshape(X_test,(-1,1000,1))
        X_valid = np.reshape(X_valid,(-1,1000,1))

        with gzip.GzipFile(path_dir+"/"+"X_train_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_train,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"X_valid_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_valid,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"X_test_"+feature_name+".pkl","wb") as fp:
            pickle.dump(X_test,fp,-1)

        with gzip.GzipFile(path_dir+"/"+"y_train_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_train,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"y_valid_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_valid,fp,-1)
        with gzip.GzipFile(path_dir+"/"+"y_test_"+feature_name+".pkl","wb") as fp:
            pickle.dump(y_test,fp,-1)

        print('export {0} flows'.format(X_train.shape[0]))
        assert  X_train.shape[0] ==len(y_train)
        assert  X_valid.shape[0] ==len(y_valid)
        assert  X_test.shape[0] ==len(y_test)
    def export_fsnet_dataset(self,path_dir,feature_name='pkt_length'):
        print('export to fsnet format')
        if os.path.exists(path_dir) == False:
            os.makedirs(path_dir)
        assert  feature_name in ['pkt_length','arv_time']
        flowCounter = 0
        for i in range(len(self.labelName)):
            ##确定名字
            package_name = self.labelName[i]
            ##提取包长
            feature_matrix =mtu * self.graphs[i].ndata[feature_name]
            fp = open(path_dir+package_name+".num","a")
            for j in range(feature_matrix.shape[0]):
                feature = ";"+"\t".join([str(int(feature_matrix[j][0][i__])) for i__ in range(feature_matrix.shape[2])])+"\t;\n"
                fp.writelines(feature)
                flowCounter+=1
            fp.close()
        print('export {0} flows'.format(flowCounter))
    @property
    def flowCounter(self):
        flowcounter = 0
        for i in range(len(self.labelName)):
            flowcounter += self.graphs[i].ndata['pkt_length'].shape[0]
        return  flowcounter

if __name__ == '__main__':

    '''#规范数据集的构建
    '''
    #构建0623 小米5plus clear数据集

    dataset = Dataset(mode='clear',
                      dumpData=True,usedumpData=False,
                      dumpFilename="D1_timecost.pkl.gzip".format(time_period),
                      cross_version=False,
                      test_split_rate=0.1,
                      graph_json_directory= r'E:\graph_neural_network_over_smartphone_application\graph_neural_network_model_v2\json_D1')
    del dataset
