from Each_EC_training import *
from Evalute import evaluate
import time
# print(len(training_data))
# 定义一个3层全连接网络，输入层有784个神经元，隐藏层30个神经元，输出层10个神经元
sizes = [784, 360, 10]
training_time=0
E=1
Max_T=21
param_agg_time=0
while E<Max_T:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #training_data是list,先随机扰动顺序，再划分成十份给所有EC。
    random.shuffle(training_data)
    training_data1=training_data[:5000]
    training_data2 = training_data[5000:10000]
    training_data3 = training_data[10000:15000]
    training_data4 = training_data[15000:20000]
    training_data5 = training_data[20000:25000]
    training_data6 = training_data[25000:30000]
    training_data7 = training_data[30000:35000]
    training_data8 = training_data[35000:40000]
    training_data9 = training_data[40000:45000]
    training_data10 = training_data[45000:]
    n_test = len(test_data)
    if E==1:
        _weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        _biases = [np.random.randn(y, 1) for y in sizes[1:]]
    else:
        param_agg_begin = time.time()
        _weights = (w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10) / 10
        #_weights = (w1 + w2 + w3)/3
        _weights = _weights.tolist()
        _biases = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b9 + b10) / 10
        #biases = (b1 + b2 + b3)/3
        #_biases = _biases.tolist()
        param_agg_end = time.time()
        param_agg_time = param_agg_end - param_agg_begin
    print("参数聚合时间：", param_agg_time)
    print("第",E,"次本地训练：")
    ID1 = '0001'
    local_trainig_begain1 = time.time()
    w1, b1, cost1 = Client(ID1, training_data1, test_data, sizes, _weights, _biases)
    w1 = np.array(w1, dtype=object)
    b1 = np.array(b1, dtype=object)
    local_trainig_end1 = time.time()
    t1 = local_trainig_end1 - local_trainig_begain1

    ID2 = '0002'
    local_trainig_begain2 = time.time()
    w2, b2, cost2 = Client(ID2, training_data2, test_data, sizes, _weights, _biases)
    w2 = np.array(w2, dtype=object)
    b2 = np.array(b2, dtype=object)
    local_trainig_end2 = time.time()
    t2 = local_trainig_end2 - local_trainig_begain2

    ID3 = '0003'
    local_trainig_begain3 = time.time()
    w3, b3, cost3 = Client(ID3, training_data3, test_data, sizes, _weights, _biases)
    w3 = np.array(w3, dtype=object)
    b3 = np.array(b3, dtype=object)
    local_trainig_end3 = time.time()
    t3 = local_trainig_end3 - local_trainig_begain3
    # print("local model trainig time1:",t3)
    #local_training_time = max(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
    local_training_time = max(t1, t2, t3)
    #print("local model trainig time:", local_training_time)

    ID4 = '0004'
    local_trainig_begain4 = time.time()
    w4, b4, cost4 = Client(ID4, training_data4, test_data, sizes, _weights, _biases)
    w4 = np.array(w4, dtype=object)
    b4 = np.array(b4, dtype=object)
    local_trainig_end4 = time.time()
    t4 = local_trainig_end4 - local_trainig_begain4
    # print("local model trainig time1:",t4)

    ID5 = '0005'
    local_trainig_begain5 = time.time()
    w5, b5, cost5 = Client(ID5, training_data5, test_data, sizes, _weights, _biases)
    w5 = np.array(w5, dtype=object)
    b5 = np.array(b5, dtype=object)
    local_trainig_end5 = time.time()
    t5 = local_trainig_end5 - local_trainig_begain5
    # print("local model trainig time1:",t5)

    ID6 = '0006'
    local_trainig_begain6 = time.time()
    w6, b6, cost6 = Client(ID6, training_data6, test_data, sizes, _weights, _biases)
    w6 = np.array(w6, dtype=object)
    b6 = np.array(b6, dtype=object)
    local_trainig_end6 = time.time()
    t6 = local_trainig_end6 - local_trainig_begain6
    # print("local model trainig time1:",t6)

    ID7 = '0007'
    local_trainig_begain7 = time.time()
    w7, b7, cost7 = Client(ID7, training_data7, test_data, sizes, _weights, _biases)
    w7 = np.array(w7, dtype=object)
    b7 = np.array(b7, dtype=object)
    local_trainig_end7 = time.time()
    t7 = local_trainig_end7 - local_trainig_begain7
    # print("local model trainig time1:", t7)

    ID8 = '0008'
    local_trainig_begain8 = time.time()
    w8, b8, cost8 = Client(ID8, training_data8, test_data, sizes, _weights, _biases)
    w8 = np.array(w8, dtype=object)
    b8 = np.array(b8, dtype=object)
    local_trainig_end8 = time.time()
    t8 = local_trainig_end8 - local_trainig_begain8
    # print("local model trainig time1:", t8)

    ID9 = '0009'
    local_trainig_begain9 = time.time()
    w9, b9, cost9 = Client(ID9, training_data9, test_data, sizes, _weights, _biases)
    w9 = np.array(w9, dtype=object)
    b9 = np.array(b9, dtype=object)
    local_trainig_end9 = time.time()
    t9 = local_trainig_end6 - local_trainig_begain9
    # print("local model trainig time1:", t9)

    ID10 = '0010'
    local_trainig_begain10 = time.time()
    w10, b10, cost10 = Client(ID10, training_data10, test_data, sizes, _weights, _biases)
    w10 = np.array(w10, dtype=object)
    b10 = np.array(b10, dtype=object)
    local_trainig_end10 = time.time()
    t10 = local_trainig_end10 - local_trainig_begain10
    # print("local model trainig time1:", t10)

    lmbda = 0.3
    print("Epoch %d: accuracy rate: %.2f%%" % (E, evaluate(test_data, _biases, _weights) / n_test * 100))
    print("cost:", cost1)
    a1 = local_training_time + param_agg_time
    training_time += a1
    print("1 Iterations训练时间：", a1, "总训练时间：", training_time)
    E += 1