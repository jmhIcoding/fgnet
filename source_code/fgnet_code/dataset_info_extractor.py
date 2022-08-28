__author__ = 'dk'
#提取数据集的一些统计信息,比如样本的数目，流的数目等等。
from graph_neural_network_model.dataset_builder import Dataset
from matplotlib import pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np
updated_apps = ['com.amazon.mShop.android.shopping', 'com.badoo.mobile', 'com.contextlogic.wish', 'com.facebook.katana', 'com.google.android.youtube', 'com.guardian', 'com.ideashower.readitlater.pro', 'com.instagram.android', 'com.kakao.talk', 'com.nytimes.android', 'com.particlenews.newsbreak', 'com.sina.weibo', 'com.snapchat.android', 'com.spotify.music', 'com.taobao.taobao', 'com.tencent.qqlivei18n', 'com.twitter.android', 'com.vidio.android', 'com.vkontakte.android', 'flipboard.app', 'ru.ideast.championat', 'tv.danmaku.bili']

def draw_node_edge_together(nodes, edges):
    ax1 = plt.axes()
    ax2 = plt.twinx()

    colors=["xkcd:bile","xkcd:bland"]
    linecolors=["xkcd:green", "xkcd:blue"]
    legends =["Node ","Edge "]
    linestyles=[".--","+--"]
    for index, datas in enumerate([nodes, edges]):
        density, bins = np.histogram(datas, bins=50, normed=True, density=True)
        density = 100.0 * density/ density.sum()
        widths = bins[:-1] - bins[1:]
        l1=ax1.bar(bins[1:],density,width=widths,color=colors[index],edgecolor="black",alpha=0.5,label= legends[index]+ "histogram")
        l2=ax2.plot(bins[1:], np.cumsum(density),linestyles[index],color= linecolors[index],alpha=0.5, label=legends[index]+ "cumulative distribution function")

    ax2.set_ylabel("Cumulative Distribution (%)",fontsize=12)
    ax1.set_xlabel("Node/Edge Number",fontsize=12)
    ax1.set_ylabel("Frequency (%)",fontsize=12)

    #ax1.legend()
    #ax2.legend()
    #plt.legend()
    ax1.legend(loc="center right",bbox_to_anchor=(0.70,0.32),fontsize=12)
    ax2.legend(loc="center right",fontsize=12)
    plt.show()

def draw_hist(datas,log=False,color="orange",linecolor="blue"):
    density, bins = np.histogram(datas, bins=50, normed=True, density=True)
    density = 100.0 * density/ density.sum()
    widths = bins[:-1] - bins[1:]
    ax1 = plt.axes()
    if log:
        l1=ax1.bar(bins[1:],np.log(density),width=widths,color=color,edgecolor="black",alpha=0.8,label="histogram")
    else:
        l1=ax1.bar(bins[1:],density,width=widths,color=color,edgecolor="black",alpha=0.5,label="histogram")
    ax1.set_xlabel("Node Number")
    ax1.set_ylabel("Frequency (%)")

    ax2 = plt.twinx()
    print(bins[1:])
    print(np.cumsum(density))

    l2=ax2.plot(bins[1:], np.cumsum(density),".--",color= linecolor,label="cumulative distribution function")
    ax2.set_ylabel("Cumulative Distribution (%)")

    #ax1.legend()
    #ax2.legend()
    #plt.legend()
    ax1.legend(loc="center right",bbox_to_anchor=(0.75,0.4))
    ax2.legend(loc="center right")
    plt.show()
def get_node_and_edges_nums(dataset_path, mode="clear"):
    dataset = Dataset(mode=mode,dumpData=False,usedumpData=True,dumpFilename=dataset_path,cross_version=False)
    print(dataset)
    nodes = []
    edges = []
    for graph in dataset.graphs:
        nodes.append(graph.number_of_nodes())
        edges.append(graph.number_of_edges())
    #draw_hist(nodes,log=False,color="xkcd:light blue",linecolor="xkcd:green")
    #draw_hist(edges,log=False,color="xkcd:light purple",linecolor="xkcd:green")
    draw_node_edge_together(nodes,edges)
def get_info(dataset_path,mode='clear'):
    dataset = Dataset(mode=mode,dumpData=False,usedumpData=True,dumpFilename=dataset_path,cross_version=False)
    self = dataset
    info = "Build {0} graph over {1} classes, {2} graph per class. {3} flow.".format(len(self.graphs),len(self.labelNameSet),len(self.graphs)//len(self.labelNameSet),self.flowCounter)
    print(info)
    print(self.class_aliasname)
    print(self.labelName)
    print(self.next_train_batch(batch_size=10))
if __name__ == '__main__':
    #get_info('D1_53_tp60.pkl.gzip')
    get_node_and_edges_nums('D1_53_tp60.pkl.gzip')