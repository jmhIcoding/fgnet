__author__ = 'dk'
#跨版本的测试
from  graph_neural_network_model import dataset_builder
from  graph_neural_network_model import spapp_classifier
from graph_neural_network_model import select_gpu
import  numpy as np
from graph_neural_network_model.model_seriealization import load
import logger_wrappers
import torch as th
from graph_neural_network_model.plotCM import plotCM
from torch import nn
from torch import optim
from torch.nn import functional as F
import tqdm
from sklearn.metrics import confusion_matrix
device_id = select_gpu.get_free_gpu_id()
use_gpu = th.cuda.is_available()
if use_gpu :
    device= "cuda:{0}".format(device_id)
else:
    device= "cpu"
#构建图加载器
data_loader = dataset_builder.Dataset(mode='clear',usedumpData=True,dumpData=True,dumpFilename="dataset_builder.crossversion.pkl.gzip")
#构建模型
model = spapp_classifier.App_Classifier(len(data_loader.labelNameSet),use_gpu=use_gpu,device=device,layer_type='GAT')
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=1e-4)
model = load(model,optimizer,checkpoint_path="./saved_model/")
if use_gpu:
    model = model.cuda(device)
    loss_func = loss_func.cuda(device)

#测试
model.eval()
batch_size = 32
acc_list =[]
real =[]
predict =[]
for subset in tqdm.trange(len(data_loader.test_index)//batch_size):
    graphs,labels = data_loader.next_test_batch(batch_size=batch_size)
    if use_gpu :
        graphs = graphs.to(th.device(device))
        labels = labels.to(th.device(device))
    predict_labels = model(graphs)
    predict_labels = F.softmax(predict_labels,1)
    argmax_labels = th.argmax(predict_labels,1)
    print(argmax_labels)
    print(labels)
    real += labels.tolist()
    predict += argmax_labels.tolist()

    acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
    acc_list.append(acc)
    info='Accuracy of argmax predictions on the test subset{1}: {0:4f}%'.format(acc,subset)
    logger_wrappers.info(info)

info = 'Average Accuracy on test set:{:0.4f}%'.format(np.mean(acc_list))
#计算混淆矩阵
conf_mat = confusion_matrix(real,predict,normalize='true')

print({'class alias name':data_loader.class_aliasname})
#class_name = [ data_loader.class_aliasname[each] for each in data_loader.class_aliasname]
class_name = [i for i in range(len(data_loader.labelNameSet))]
plotCM(class_name,conf_mat,'crossversion_confusion_matrix.png')
logger_wrappers.info(info)