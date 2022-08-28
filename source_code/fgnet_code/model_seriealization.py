__author__ = 'dk'
#模型的保存和加载
import torch
import json
import logger_wrappers
import os
model_name = "/gnn_model.pkl"
def save(model,optimizer,checkpoint_path, model_name= model_name):
    path = checkpoint_path + model_name
    if os.path.exists(checkpoint_path) == False:
        os.makedirs(checkpoint_path)
    torch.save(model,path)
    #torch.save(
    #    {'state_dict':model.state_dict(),
    #     'optimizer':optimizer.state_dict()},
    #    (checkpoint_path+model_name).replace("//","/")
    #)
    info = "Dump model to {0} well.".format(checkpoint_path)
    logger_wrappers.warning(info)

def load(model,optimizer,checkpoint_path, model_name = model_name):
    path = (checkpoint_path+model_name).replace("//","/")
    if os.path.exists(path):
        model_CKPT = torch.load(path)
        model = model_CKPT
        #model.load_state_dict(model_CKPT['state_dict'])
        #optimizer.load_state_dict(model_CKPT['optimizer'])
        info ="Load model from {0} well.".format(path)
        logger_wrappers.warning(info)
    else:
        logger_wrappers.warning('Load empty model from {0}.'.format(path))
    return model#,optimizer
