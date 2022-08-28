__author__ = 'dk'
from  graph_neural_network_model import dataset_builder
from  graph_neural_network_model import spapp_classifier
from graph_neural_network_model import select_gpu
import  numpy as np
from graph_neural_network_model.model_seriealization import save,load
import logger_wrappers
import torch as th
import matplotlib.pylab as plt
from torch import nn
from torch import optim
from torch.nn import functional as F
import tqdm
device_id = select_gpu.get_free_gpu_id()
use_gpu = th.cuda.is_available()
if use_gpu :
    device= 2#"cuda:{0}".format(device_id)
else:
    device= "cpu"
def main(trainset,modelpath="./saved_model/",layer_type='GAT',max_epoch=60):
    #构建图加载器
    data_loader = dataset_builder.Dataset(mode='clear',usedumpData=True,dumpData=True,dumpFilename=trainset)
    #构建模型
    model = spapp_classifier.App_Classifier(len(data_loader.labelNameSet),use_gpu=use_gpu,device=device,layer_type=layer_type)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=5e-5)
    model = load(model,optimizer,checkpoint_path=modelpath)
    if use_gpu:
        model = model.cuda(device)
        loss_func = loss_func.cuda(device)

    #训练
    model.train()
    epoch_losses = []
    epoch_acces = []
    batch_size = 1

    ##最后完整的测试一波

    model.eval()
    acc_list =[]
    y_real = []
    y_pred = []
    calculated_graphs = []
    for subset in range(len(data_loader.test_index)//batch_size)[:200]:
        graphs,labels = data_loader.next_test_batch(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(th.device(device))
            labels = labels.to(th.device(device))
        predict_labels = model(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = th.argmax(predict_labels,1)
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        acc_list.append(acc)
        info='Accuracy of argmax predictions on the test subset{1}: {0:4f}%'.format(acc,subset)
        logger_wrappers.info(info)
        y_real +=labels.tolist()
        y_pred += argmax_labels.tolist()
        if use_gpu :
            calculated_graphs.append(graphs.to(th.device('cpu')))

    import pickle
    with open('calculated_graphs.pkl','wb') as fp:
        pickle.dump(calculated_graphs, fp)

    from accuracy_per_class import accuracy_per_class
    accuracy_per_class(y_real,y_pred)
    info = 'Average Accuracy on test set:{:0.4f}%'.format(np.mean(acc_list))
    logger_wrappers.info(info)
if __name__ == '__main__':
    main(trainset="D1_53_tp60_fc.pkl.gzip",
         modelpath="saved_model_fc",
         layer_type="GAT",
         max_epoch=0)

