Dataset for `Accurate Mobile-App Fingerprinting Using Flow-level Relationship with Graph Neural Networks`.
# This work is being accepted by Computer Networks.
# Background of this dataset
We collected another private encrypted mobile application traffic dataset across weeks, to evaluate the generalization of our method in dealing with ambiguous traffic and the performance against traffic concepts drift.
Here go the details of the dataset setup:

 - Equipment setup
  As Figure-3 indicates, smartphones with apps communicate with the Internet via a WiFi access point (AP) and the AP forwards the packets into two gateways that come from different ISPs.
  To generate traffic from apps, we used scripts that communicated with the target mobile via USB using Android Debug Bridge (ADB).
  These scripts were sent by the controller computer, and mainly contained UI commands that simulated user actions within apps and system commands that configured the devices.
  ![Figure-3](https://img-blog.csdnimg.cn/6219b825d7334c99bfaa09654260f3e8.png)

- Applications selection
  We selected 53 apps from the apps list used by AppScanner after filtering out several plain-text applications which consist of a relatively low fraction of encrypted traffic. These apps come from different regions such as shopping, magazines, social and so on. We always installed the latest versions onto the selected devices and signed up for each app.
  
- Network trace collection
  We cyclically performed UI fuzzing operations on each app which is activated for about 30 seconds every time via monkeyrunner as Appscanner and passively collected the network traffic between smartphones and the AP. To collect pure network traces from specific apps, we configured the android devices with the iptables rules and listened on the NFLOG. We also filtered out retransmit, out of order and zero-payload packets.

We collected our private dataset from 23rd June and obtained a dataset named D1. Then we collected dataset D2 one month later after the D1 was captured, and 22 apps have been updated.

# Directory Structure
We have extracted the side-channel informantion such as packet size, packet arrival time from the raw pcap files and shared them in the format of json file.

All the json files are layouted as `dataset/{datasetName}/{appName}/{appVersion}/{timestamp}_clear.pcap.json`.
And each json file 
Let us take an example, for a pcap sample for `bbc.mobile.weather` of version 4.0.6, the parsed json file is:
`dataset/D1/bbc.mobile.weather/4.0.6/1592638931_clear.pcap.json`.

# Data structure for each Json file
For each parsed json file, it contains all the neccessary side-channel information of all the network flows in its corresponding pcap file.

Here is an example: there are six network flows in the corresponding pcap, so there are six items within the list. For each item, we present the packet length sequence, arrive_time_delta, start_timestamp of flow, end_timestamp of flow and src_port.
```bash
[
  {
    "packet_length": [
      282,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      475
    ],
    "payload": [],
    "arrive_time_delta": [
      0,
      0.0000030994415283203125,
      0.0002589225769042969,
      0.00014519691467285156,
      0.0002589225769042969,
      0.00004792213439941406,
      0.0001900196075439453,
      0.0000030994415283203125,
      0.0017788410186767578,
      0.000010013580322265625,
      0.00045800209045410156,
      0.0000059604644775390625,
      0.000014066696166992188,
      0.00005698204040527344,
      0.000013113021850585938,
      0.000033855438232421875,
      0.0047261714935302734,
      0.000003814697265625,
      0.00006699562072753906,
      0.0000030994415283203125,
      0.03981494903564453
    ],
    "start_timestamp": 1592747812.358703,
    "end_timestamp": 1592747812.406596,
    "src_port": 38075
  },
  {
    "packet_length": [
      178,
      -1448,
      -600,
      -792,
      -1428,
      -1021,
      126,
      53,
      44,
      42,
      -327,
      870,
      -38,
      38,
      -728,
      -77,
      31
    ],
    "payload": [],
    "arrive_time_delta": [
      0,
      0.08965301513671875,
      0.00012612342834472656,
      0.0000059604644775390625,
      0.0989830493927002,
      0.0001308917999267578,
      0.0000059604644775390625,
      0.2170701026916504,
      0.000015020370483398438,
      0.0000050067901611328125,
      0.0001468658447265625,
      0.0000059604644775390625,
      0.0000050067901611328125,
      0.0000030994415283203125,
      0.2625288963317871,
      13.7536141872406,
      0.000007867813110351562
    ],
    "start_timestamp": 1592747813.855156,
    "end_timestamp": 1592747828.277463,
    "src_port": 43551
  },
  {
    "packet_length": [
      174,
      -1424,
      -1440,
      -1448,
      -600,
      -347,
      126,
      53,
      86,
      1448,
      524,
      -327,
      -38,
      38,
      1448,
      1415,
      1448,
      908,
      1448,
      328
    ],
    "payload": [],
    "arrive_time_delta": [
      0,
      0.0028209686279296875,
      0.00014710426330566406,
      0.022533893585205078,
      0.0001399517059326172,
      0.0000059604644775390625,
      0.06827020645141602,
      0.0000059604644775390625,
      0.00012803077697753906,
      0.08455181121826172,
      0.000014066696166992188,
      0.0001780986785888672,
      0.0000059604644775390625,
      0.05992698669433594,
      0.05175304412841797,
      0.00013899803161621094,
      1.9377648830413818,
      0.00013017654418945312,
      2.415329933166504,
      0.00000286102294921875
    ],
    "start_timestamp": 1592747814.759279,
    "end_timestamp": 1592747819.403128,
    "src_port": 37249
  },
  {
    "packet_length": [
      174,
      -1448,
      -600,
      -816,
      -1440,
      -955,
      126,
      53,
      86,
      -327
    ],
    "payload": [],
    "arrive_time_delta": [
      0,
      0.14483189582824707,
      0.0002219676971435547,
      0.008244037628173828,
      0.000011920928955078125,
      0.036933183670043945,
      0.00014090538024902344,
      0.0000050067901611328125,
      0.0000059604644775390625,
      0.411592960357666
    ],
    "start_timestamp": 1592747814.853192,
    "end_timestamp": 1592747815.45518,
    "src_port": 37250
  },
  {
    "packet_length": [
      218,
      -1448,
      -600,
      -750,
      93,
      -361,
      93,
      38,
      436,
      -38,
      -1070,
      -1440,
      -1378,
      -1440,
      -756,
      -268,
      46
    ],
    "payload": [],
    "arrive_time_delta": [
      0,
      0.010441064834594727,
      0.00015211105346679688,
      0.0000059604644775390625,
      0.17130398750305176,
      0.000011920928955078125,
      0.0000040531158447265625,
      0.0000019073486328125,
      0.0001289844512939453,
      0.000008106231689453125,
      0.00156402587890625,
      0.000011920928955078125,
      0.00012612342834472656,
      0.0000050067901611328125,
      0.6461899280548096,
      0.00015592575073242188,
      0.0000059604644775390625
    ],
    "start_timestamp": 1592747823.029281,
    "end_timestamp": 1592747823.859398,
    "src_port": 44086
  },
  {
    "packet_length": [
      282,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      1448,
      764
    ],
    "payload": [],
    "arrive_time_delta": [
      0,
      0.00006008148193359375,
      0.000102996826171875,
      0.000010967254638671875,
      0.00003695487976074219,
      0.00008392333984375,
      0.0000030994415283203125,
      0.00009202957153320312,
      0.00000286102294921875,
      0.010506153106689453,
      0.000012874603271484375,
      0.000016927719116210938,
      0.0002791881561279297,
      0.000011920928955078125,
      0.0000400543212890625,
      0.000009059906005859375,
      0.0000059604644775390625,
      0.00003790855407714844,
      0.0000050067901611328125,
      0.00003695487976074219,
      0.0000050067901611328125,
      0.011543989181518555,
      0.0001361370086669922
    ],
    "start_timestamp": 1592747829.196964,
    "end_timestamp": 1592747829.220004,
    "src_port": 38255
  }
]
```

Currently, we only open the side-channel information of our dataset for  convenience.

If you want to get the raw pcap files of our dataset, please contact us with the following E-mail: jiangminghao@iie.ac.cn, gougaopeng@iie.ac.cn, caiwei@iie.ac.cn.
