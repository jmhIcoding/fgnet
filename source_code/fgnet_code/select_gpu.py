__author__ = 'jmh081701'
import os
import  sys
import numpy as np
cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv"
def get_free_gpu_id():
    pipe =os.popen(cmd)
    freeMbs =[]
    for eachline in pipe:
        if 'index' in eachline:
            continue
        id,freeMb=eachline.split(',')
        freeMb = int(freeMb.replace("MiB",""))
        freeMbs.append(freeMb)
    if len(freeMbs)== 0 :
        return "0"
    return str(np.argmax(freeMbs))
def set_visible_gpu():
    if sys.platform=='linux':
        os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu_id()
    else:
        os.environ['CUDA_VISBALE_DEIVCES'] ='0'
if __name__ == '__main__':
    print(get_free_gpu_id())