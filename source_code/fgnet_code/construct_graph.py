__author__ = 'dk'
#构建图
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import  torch as th
import copy
from request_logit import get_logit
import random
import logger_wrappers as log
pad_length = 1000           #包长序列和包到达时间序列的填充(截断)长度
mtu = 1500                  #MTU数值,用于归一化包长序列
mtime = 10                  #达到时间间隔最大数值,用于归一化时间序列
def pad_sequence(source,pad_length,pad_value):
    if len(source) > pad_length:
        return  source[:pad_length]
    else:
        source = source + [pad_value] * (pad_length - len(source))
        return  source

def build_graph(sample):
    graph = dgl.DGLGraph()
    graph.add_nodes(len(sample['nodes']))
    #添加边的属性
    pkt_length_matrix  = np.zeros(shape=(len(sample['nodes']),pad_length),dtype =np.double)
    arv_time_matrix = np.zeros(shape=(len(sample['nodes']),pad_length),dtype = np.double)
    for i in range(len(sample['nodes'])):
        pkt_length_matrix[i] = pad_sequence(sample['nodes'][i]['packet_length'],pad_length,0)
        arv_time_matrix[i] = pad_sequence(sample['nodes'][i]['arrive_time_delta'],pad_length,0)
    pkt_length_matrix = np.reshape(pkt_length_matrix,(-1,1,pad_length))
    arv_time_matrix = np.reshape(arv_time_matrix,(-1,1,pad_length))
    graph.ndata['pkt_length'] = pkt_length_matrix / mtu        #顺带完成归一化
    graph.ndata['arv_time'] = arv_time_matrix / mtime

    graph.ndata['burst_id'] = np.array([node['burst_id'] for node in sample['nodes']])

    for each in sample['edges']:
        graph.add_edge(each[0],each[1])
    graph.edata['attn'] = np.random.randn(len(graph.edges),1)

    return  graph

#区块链数据集
# sni_blacklist=[
# 'fonts.googleapi.com',
# 'www.google-analytics.com',
# 'fonts.gstatic.com',
# 'stats.g.doubleclick.net',
# 'www.googletagmanager.com',
# 'mainnet.infura.io',
# 'cdn.conctentful.com',
# 'use.typekit.net',
# 'cdnjs.cloudflare.com',
# 'api.thegraph.com'
# ]
sni_blacklist = []
concurrent_time_threshold = 1
def build_graphs_from_json(filename, time_period, concurrent_time_threshold=concurrent_time_threshold):
    log.info('Load graph from {0}'.format(filename))
    with open(filename) as fp:
        sample = json.load(fp)
    try:
        if len(sample) == 0:
            raise BaseException('Empty graph!')
        #for i in range(len(sample)):
        #    size = [abs(x) for x in sample[i]['packet_length']]
        #    sample[i]['fsnet']=get_logit([size])

        for each in sample:
            if 'start_timestamp' not in each:
                each['start_timestamp'] = min(each['timestamp'])
            if 'arrive_time_delta' not in each :
                each['arrive_time_delta'] = [0] + [each['timestamp'][i]- each['timestamp'][i-1] for i in range(1, len(each['timestamp']))]
        sample.sort(key=lambda x : x['start_timestamp'])
        _graphs = []
        graphs = []
        for i in range(len(sample)):
            if 'sni' in sample[i] and sample[i]['sni'] in sni_blacklist:
                continue

            if len(sample[i]['arrive_time_delta']) < 10:
                continue

            if len(_graphs)== 0 :
                _graphs.append([sample[i]])
            else:
                if sample[i]['start_timestamp'] - _graphs[-1][0]['start_timestamp'] < time_period :
                    _graphs[-1].append(sample[i])
                else:
                    _graphs.append([sample[i]])
        for S in _graphs:
            nodes = []
            edges = []
            burst = []
            #burst = [copy.deepcopy(S[0])]
            #burst[0]['start_timestamp']=0
            #burst[0]['id']=0
            #nodes.append(burst[0])
            last_burst = burst

            bursts = []
            for index, flow in enumerate(S):
                flow['id'] = index#index+1

                if len(burst) == 0:
                    burst.append(flow)
                else:
                    if abs(flow['start_timestamp']-burst[-1]['start_timestamp']) < concurrent_time_threshold:
                        burst.append(flow)
                    else:
                        ##burst内部的连接
                        for j in range(len(burst)-1):
                            for k in range(j+1, len(burst)):
                                ##burst内部也全连接
                                #edges.append((burst[j]['id'],burst[k]['id']))

                                #不要全连接
                                edges.append((burst[j]['id'],burst[k]['id']))
                                break
                        #burst之间的连接

                        if len(last_burst)> 0:
                            #1.论文的连接
                            edges.append((last_burst[-1]['id'],burst[0]['id']))
                            if len(burst)>1:
                               edges.append((last_burst[-1]['id'],burst[-1]['id']))

                            #2.全连接
                            # for last in last_burst:
                            #       for current in burst:
                            #           edges.append((last['id'],current['id']))

                            #3.随机连接
                            # random_edge_up = random.randint(1, len(last_burst) * len(burst))
                            # while random_edge_up > 0 :
                            #     i = random.randint(last_burst[0]['id'],last_burst[-1]['id'])
                            #     j = random.randint(burst[0]['id'],  burst[-1]['id'])
                            #     edges.append((i,j))
                            #     random_edge_up -= 1


                            #4. burst的首尾相连
                            # edges.append((last_burst[-1]['id'],burst[0]['id']))
                            # if len(last_burst) > 1:
                            #    edges.append((last_burst[0]['id'],burst[0]['id']))
                            # if len(burst)>1:
                            #    edges.append((last_burst[-1]['id'],burst[-1]['id']))
                            #    if len(last_burst) > 1:
                            #        edges.append((last_burst[0]['id'],burst[-1]['id']))
                        #print(burst[0]['id'],burst[0]['start_timestamp'],burst[-1]['id'], burst[-1]['start_timestamp'])
                        last_burst = burst
                        burst = [flow]
                        bursts.append(last_burst)
                flow['burst_id'] = len(bursts)
                nodes.append(flow)
            if len(burst) > 0:
                for j in range(len(burst)-1):
                    for k in range(j+1, len(burst)):
                         ##burst内部也全连接
                         #edges.append((burst[j]['id'],burst[k]['id']))

                         #不要全连接
                         edges.append((burst[j]['id'],burst[k]['id']))
                         break
                if len(last_burst)> 0:
                    #1.论文的连接
                    edges.append((last_burst[-1]['id'],burst[0]['id']))
                    if len(burst)>1:
                      edges.append((last_burst[-1]['id'],burst[-1]['id']))

                    #2.全连接
                    # for last in last_burst:
                    #      for current in burst:
                    #         edges.append((last['id'],current['id']))

                    #3.随机连接
                    # random_edge_up = random.randint(1, len(last_burst) * len(burst))
                    # while random_edge_up > 0 :
                    #     i = random.randint(last_burst[0]['id'],last_burst[-1]['id'])
                    #     j = random.randint(burst[0]['id'],  burst[-1]['id'])
                    #     edges.append((i,j))
                    #     random_edge_up -=1


                     #4. burst的首尾相连
                      # edges.append((last_burst[-1]['id'],burst[0]['id']))
                      # if len(last_burst) > 1:
                      #    edges.append((last_burst[0]['id'],burst[0]['id']))
                      # if len(burst)>1:
                      #    edges.append((last_burst[-1]['id'],burst[-1]['id']))
                      #    if len(last_burst) > 1:
                      #        edges.append((last_burst[0]['id'],burst[-1]['id']))
                bursts.append(burst)
                #print(burst[0]['id'],burst[0]['start_timestamp'],burst[-1]['id'], burst[-1]['start_timestamp'])
            graph = build_graph({
                'nodes': nodes,
                'edges':edges
            })
            graphs.append(graph)
        if len(graphs)>1 :
            if abs(_graphs[-1][-1]['start_timestamp'] - _graphs[-1][0]['start_timestamp']) < time_period:
                graphs.pop(-1)
        return  graphs
    except BaseException as exp:
            #如果节点数为0,就不构图
            info='build ill graph from {0}'.format(filename)
            log.warning(info)
            log.error(exp)
            #raise exp
            return [None]

if __name__ == '__main__':
    gs = build_graphs_from_json(
        r'D:\social_webpage_attack\dataset\weibo_json\D1\234371797\1633062760.pcap.json',
        time_period=60
    )[0]

    gs2 =  build_graphs_from_json(
        r'D:\social_webpage_attack\dataset\weibo_json\D1\234371797\1633062760.pcap.json',
                time_period=60
    )[0]


    import pickle
    with open('graphs.pkl','wb') as fp:
        pickle.dump([gs,gs2],fp)

    nx_G = gs.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True)
    plt.show()