__author__ = 'dk'
#Smartphone Application Classifier
import  torch as th
import  torch.nn as nn
import  torch.nn.functional as F
import dgl
from torchsummary import summary
from graph_neural_network_model.feature_extractor import Deep_fingerprinting as Extractor
from graph_neural_network_model.feature_extractor import IDENTIFY as Extractor_idt
from graph_neural_network_model.feature_extractor import AWF as Extractor_awf
from dgl.nn.pytorch import GraphConv,GATConv
class App_Classifier(nn.Module):
    def __init__(self,nb_classes=55,nb_layers=2,latent_feature_length=100,use_gpu=False,device="cpu",layer_type='GCN'):
        assert  layer_type in ['GCN','GAT']
        print('Train {0} model.'.format(layer_type))
        super(App_Classifier,self).__init__()
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        self.layer_type = layer_type
        self.latent_feature_length = latent_feature_length

        #self.pkt_length_fextractor = Extractor(self.latent_feature_length)
        #self.arv_time_fextractor = Extractor(self.latent_feature_length)
        self.fs_fextractor = Extractor(self.latent_feature_length)
        if use_gpu:
            #self.arv_time_fextractor = self.arv_time_fextractor.cuda(device)
            #self.pkt_length_fextractor = self.pkt_length_fextractor.cuda(device)
            self.fs_fextractor = self.fs_fextractor.cuda(device)
        self.use_gpu = use_gpu
        self.device = device
        self.layers=[]
        head_nums = [1] + [int(1.5**(nb_layers-i)) for i in range(nb_layers)]
        for i in range(nb_layers):
            if layer_type =='GCN':
                #图卷积模型
                print('Build GCN : in_feats={0}，out_feats={1}'.format(self.latent_feature_length * int(1.6**i),self.latent_feature_length * int(1.6**(i+1))))
                layer = GraphConv(in_feats=self.latent_feature_length * int(1.6**i),out_feats=self.latent_feature_length * int(1.6**(i+1)))
            elif layer_type =='GAT':
                print('Build GAT : in_feats={0}，out_feats={1},num_heads={2}'.format(self.latent_feature_length * int(1.6**i)*head_nums[i],self.latent_feature_length  * int(1.6**(i+1)),head_nums[i+1]))
                layer = GATConv(in_feats=self.latent_feature_length * int(1.6**i) *head_nums[i],out_feats=self.latent_feature_length  * int(1.6**(i+1)),num_heads=head_nums[i+1])
            if use_gpu :
                layer = layer.to(th.device(device))
            self.layers.append(layer)
        self.classify = nn.Linear(in_features=self.latent_feature_length * int(1.6 **nb_layers),out_features=nb_classes)


    def forward(self, g):
        ##特征提取
        #pkt_length_matrix  = self.pkt_length_fextractor(g.ndata['pkt_length'].float())
        #arv_time_matrix = self.arv_time_fextractor(g.ndata['arv_time'].float())
        fsnet_matrix = self.fs_fextractor(g.ndata['fsnet']).float()
        ##图卷积
        for layer in self.layers:
            #pkt_length_matrix = layer(g,pkt_length_matrix.to(th.device(self.device)))
            #arv_time_matrix = layer(g,arv_time_matrix.to(th.device(self.device)))
            fsnet_matrix = layer(g,fsnet_matrix.to(th.device(self.device)))
            if self.layer_type =='GAT':
                #pkt_length_matrix = th.flatten(pkt_length_matrix,1)
                #arv_time_matrix= th.flatten(arv_time_matrix,1)
                fsnet_matrix = th.flatten(fsnet_matrix,1)
        #g.ndata['pkt_length'] = pkt_length_matrix
        #g.ndata['arv_time'] = arv_time_matrix
        g.ndata['fsnet'] = fsnet_matrix
        #pkt_length_matrix = dgl.mean_nodes(g,'pkt_length')
        #arv_time_matrix = dgl.mean_nodes(g,'arv_time')
        fsnet_matrix= dgl.mean_nodes(g,'fsnet')
        #matrix = th.cat((pkt_length_matrix,arv_time_matrix),1)
        return  self.classify(fsnet_matrix)

