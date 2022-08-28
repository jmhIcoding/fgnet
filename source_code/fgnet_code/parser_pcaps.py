__author__ = 'dk'
import os
import sys
import shutil
import json
import threading
import time
import flowcontainer
import tqdm
sni_blacklist = {'accounts.google.com',
                 'apis.google.com',
                 'auth.grammarly.com',
                 'content-autofill.googleapis.com',
                 'data.grammarly.com',
                 'gnar.grammarly.com',
                 'lh3.googleusercontent.com',
                 'mtalk.google.com',
                 'play.google.com',
                 'treatment.grammarly.com',
                 'www.google.com',
                 'www.googletagmanager.com',
                 'www.gstatic.com',
                 }
sni_blacklist=[]
semaphere = threading.Semaphore(8) #并发数目,8就可以让CPU使用率100%
def get_flows(src_pcap):
    flows = flowcontainer.extractor.extract(src_pcap, extension='tls.handshake.extensions_server_name')
    result = []
    for each in flows:
        arrive_time_delta =[]
        if len(flows[each].lengths) < 2:
            continue
        if len(flows[each].timestamps) == 0:
            continue
        for _index in range(len(flows[each].timestamps)):
            if len(arrive_time_delta) ==0 :
                arrive_time_delta.append(flows[each].timestamps[_index])
            else:
                arrive_time_delta.append(flows[each].timestamps[_index]-flows[each].timestamps[_index-1])
        arrive_time_delta[0]=0.0
        if 'tls.handshake.extensions_server_name' in flows[each].extension:
            sni = flows[each].extension['tls.handshake.extensions_server_name'][0][0]
        else:
            sni = ''

        if sni in sni_blacklist:
            continue

        result.append({
            'packet_length':flows[each].lengths,
            "payload":[],
            "arrive_time_delta":arrive_time_delta,
            "start_timestamp":flows[each].time_start,
            "end_timestamp":flows[each].time_end,
            'src_port':flows[each].sport,
            'dst_ip':flows[each].dst,
            'dst_port':flows[each].dport,
            'sni':sni}
        )

    return  result
def dump_json(pcap_name,json_name,file):
    try:
        flows= get_flows(pcap_name)
    except BaseException as exp:
        print(exp)
        semaphere.release()
        raise exp
        return
    if not os.path.exists(json_name):
        os.makedirs(json_name)
    with open(json_name+file+".json",'w') as fp:
        json.dump(flows, fp)
    semaphere.release()
if __name__ == '__main__':
    pcap_src_directory = r"E:\graph_neural_network_over_smartphone_application\Traffic_generator\pcaps"
    json_diretory = "./json_D1_ipaddress/"
    pcap_names = []
    json_names = []
    packageNames = []
    files = []
    for _root,_dirs,_files in os.walk(pcap_src_directory):
        if  len(_files)==0:
            continue
        #versionName = _root.split("\\")[-1]
        packageName = _root.split("\\")[-2]
        for file in _files:
            pcap_name = _root +"\\" + file
            json_name = json_diretory+"/"+packageName+"/"+"/"
            if 'clear' not in file:
                 continue
            #if os.path.exists(json_name+file+".json") == True:
            #    print(json_name+file+".json"," exist")
            #    continue
            #if os.stat(pcap_name).st_size < 2* (1024 ** 2):
            #    continue
            packageNames.append(packageName)
            json_names.append(json_name)
            pcap_names.append(pcap_name)
            files.append(file)
    import random
    index_list =  [x for x in range(len(packageNames))]
    random.shuffle(index_list)

    for index in tqdm.tqdm(index_list):
            pcap_name = pcap_names[index]
            json_name = json_names[index]
            file  = files[index]

            semaphere.acquire()
            th = threading.Thread(target=dump_json,args=(pcap_name, json_name, file))
            time.sleep(0.2)
            th.start()
