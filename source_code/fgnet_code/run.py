__author__ = 'dk'
##训练脚本
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
    device= "cuda:{0}".format(device_id)
else:
    device= "cpu"

#构建模型
#参数
parameter={
    'E1':{
        'latent_feature_length':40,
        'nb_layer':2
    },
        'E2':{
        'latent_feature_length':60,
        'nb_layer':2
    },
        'E3':{
        'latent_feature_length':80,
        'nb_layer':2
    },
        'E4':{
        'latent_feature_length':100,
        'nb_layer':2
    },
        'E5':{
        'latent_feature_length':120,
        'nb_layer':2
    },
        'E6':{
        'latent_feature_length':140,
        'nb_layer':2
    },
        'E7':{
        'latent_feature_length':200,
        'nb_layer':2
    },
        'E8':{
        'latent_feature_length':300,
        'nb_layer':2
    },
        'E9':{
        'latent_feature_length':100,
        'nb_layer':1
    },
        'E10':{
        'latent_feature_length':100,
        'nb_layer':3
    },
            'E11':{
        'latent_feature_length':100,
        'nb_layer':4
    },
            'E12':{
        'latent_feature_length':100,
        'nb_layer':5
    },
            'E13':{
        'latent_feature_length':100,
        'nb_layer':6
    },
}
finished={'E1'}
for each in parameter:
    if each  in finished:
        continue
    print('#'*100)
    print('Now begin new experiment!')
    print(parameter[each])
#构建图加载器
    data_loader = dataset_builder.Dataset(mode='clear',usedumpData=True,dumpData=False,dumpFilename="/home3/jmh/D1_53.pkl.gzip")
    model = spapp_classifier.App_Classifier(len(data_loader.labelNameSet),use_gpu=use_gpu,device=device,layer_type='GAT',
                                            latent_feature_length=parameter[each]['latent_feature_length'],
                                            nb_layers=parameter[each]['nb_layer'])

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=1e-4)
    model = load(model,optimizer,checkpoint_path="./saved_model_{0}/".format(each))
    if use_gpu:
        model = model.cuda(device)
        loss_func = loss_func.cuda(device)

    #训练
    model.train()
    epoch_losses = []
    epoch_acces = []
    max_epoch = 60
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
        save(model,optimizer,checkpoint_path="./saved_model_{0}/".format(each))
    if len(epoch_acces)!=0 and len(epoch_losses)!=0:
        plt.title('loss and accuracy across epoches')
        plt.plot(epoch_losses,label='loss')
        plt.plot(epoch_acces,label='accuracy')
        plt.savefig("./epoch_losses.png")
    ##最后完整的测试一波

    model.eval()
    acc_list =[]
    for subset in range(len(data_loader.test_index)//batch_size):
        graphs,labels = data_loader.next_test_batch(batch_size=batch_size)
        if use_gpu :
            graphs = graphs.to(th.device(device))
            labels = labels.to(th.device(device))
        predict_labels = model(graphs)
        predict_labels = F.softmax(predict_labels,1)
        argmax_labels = th.argmax(predict_labels,1)
        print(argmax_labels)
        print(labels)
        acc = (labels == argmax_labels).float().sum().item() / len(labels) * 100
        acc_list.append(acc)
        info='Accuracy of argmax predictions on the test subset{1}: {0:4f}%'.format(acc,subset)
        logger_wrappers.info(info)
    info = 'Average Accuracy on test set:{:0.4f}%'.format(np.mean(acc_list))
    logger_wrappers.info(info)

    del model
    del data_loader