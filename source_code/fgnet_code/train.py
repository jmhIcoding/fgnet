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
    device= 0#"cuda:{0}".format(device_id)
else:
    device= "cpu"
def main(trainset,modelpath="./saved_model/",layer_type='GAT',max_epoch=60):
    #构建图加载器
    data_loader = dataset_builder.Dataset(mode='clear',usedumpData=True,dumpData=True,dumpFilename=trainset)
    #构建模型
    model = spapp_classifier.App_Classifier(len(data_loader.labelNameSet),use_gpu=use_gpu,device=device,layer_type=layer_type)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=1e-4)
    model = load(model,optimizer,checkpoint_path=modelpath)
    if use_gpu:
        model = model.cuda(device)
        loss_func = loss_func.cuda(device)

    #训练
    model.train()
    epoch_losses = []
    epoch_acces = []
    batch_size = 16
    for epoch in tqdm.trange(max_epoch):
        epoch_loss = 0
        iter = 0
        while data_loader.epoch_over == epoch:
            graphs,labels= data_loader.next_train_batch(batch_size)
            if use_gpu :
                graphs = graphs.to(th.device(device))
                labels = labels.to(th.device(device))
            predict_label = model(graphs)
            #print(predict_label.size())
            #print(labels.size())
            loss = loss_func(predict_label,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_gpu:
                lv= loss.detach().item()
            else:
                lv = loss.detach().cpu().item()
            epoch_loss += lv
            iter +=1
            #print('Inner loss: {:.4f},Train Watch:{}'.format(lv,data_loader.train_watch))
            #epoch_losses.append(lv)
        epoch_loss /= (iter+0.0000001)
        info='Epoch {}, loss: {:.4f}'.format(epoch,epoch_loss)
        logger_wrappers.warning(info)
        epoch_losses.append(epoch_loss)
        #测试一下:
        graphs,labels = data_loader.next_valid_batch(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(th.device(device))
            labels = labels.to(th.device(device))
        predict_labels = model(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = th.argmax(predict_labels,1)
        print(argmax_labels)
        print(labels)
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        info='Accuracy of argmax predictions on the valid set: {:4f}%'.format(
            acc)
        epoch_acces.append(acc)
        logger_wrappers.info(info)
        ###保存一下模型
        save(model,optimizer,checkpoint_path=modelpath)

    ##最后完整的测试一波

    model.eval()
    acc_list =[]
    y_real = []
    y_pred = []

    for subset in range(len(data_loader.test_index)//batch_size):
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
    from accuracy_per_class import accuracy_per_class
    accuracy_per_class(y_real,y_pred)
    info = 'Average Accuracy on test set:{:0.4f}%'.format(np.mean(acc_list))
    logger_wrappers.info(info)
if __name__ == '__main__':
    main(trainset="D1_53_tp60.pkl.gzip",
         modelpath="saved_model_tp60",
         layer_type="GAT",
         max_epoch=0)

