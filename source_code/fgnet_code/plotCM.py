__author__ = 'dk'
#绘制混淆矩阵
import  numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(20,20),dpi=600)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.1f' % (matrix[i, i] * 100)), va='center', ha='center',fontsize=8)
    ax.set_xticklabels([''] + classes, rotation=90,fontsize=12)
    ax.set_yticklabels([''] + classes,fontsize=12)
    #save
    plt.savefig(savname)