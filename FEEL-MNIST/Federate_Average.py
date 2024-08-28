from EC_training import *
from Evalute import evaluate
# print(len(training_data))
# 定义一个3层全连接网络，输入层有784个神经元，隐藏层30个神经元，输出层10个神经元
sizes = [784, 360, 10]
E=1
Max_T=21
while E<Max_T:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    #training_data是list,先随机扰动顺序，再划分成十份给所有客户。
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
        _weights=(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10)/10
        _weights=0.2*weights+0.8*_weights
        _biases = (b1+b2+b3+b4+b5+b6+b7+b9+b10)/10
        _biases = 0.2*biases+0.8*_biases
    print("第",E,"次本地训练：")
    ID1 = '0001'
    w1, b1, cost1 = Client(ID1, training_data1, test_data, sizes, _weights, _biases)
    w1 = np.array(w1, dtype=object)
    b1 = np.array(b1, dtype=object)
    ID2 = '0002'
    w2, b2, cost2 = Client(ID2, training_data2, test_data, sizes, _weights, _biases)
    w2 = np.array(w2, dtype=object)
    b2 = np.array(b2, dtype=object)
    ID3 = '0003'
    w3, b3, cost3 = Client(ID3, training_data3, test_data, sizes, _weights, _biases)
    w3 = np.array(w3, dtype=object)
    b3 = np.array(b3, dtype=object)
    ID4 = '0004'
    w4, b4, cost4 = Client(ID4, training_data4, test_data, sizes, _weights, _biases)
    w4 = np.array(w4, dtype=object)
    b4 = np.array(b4, dtype=object)
    ID5 = '0005'
    w5, b5, cost5 = Client(ID5, training_data5, test_data, sizes, _weights, _biases)
    w5 = np.array(w5, dtype=object)
    b5 = np.array(b5, dtype=object)
    ID6 = '0006'
    w6, b6, cost6 = Client(ID6, training_data6, test_data, sizes, _weights, _biases)
    w6 = np.array(w6, dtype=object)
    b6 = np.array(b6, dtype=object)
    ID7 = '0007'
    w7, b7, cost7 = Client(ID7, training_data7, test_data, sizes, _weights, _biases)
    w7 = np.array(w7, dtype=object)
    b7 = np.array(b7, dtype=object)
    ID8 = '0008'
    w8, b8, cost8 = Client(ID8, training_data8, test_data, sizes, _weights, _biases)
    w8 = np.array(w8, dtype=object)
    b8 = np.array(b8, dtype=object)
    ID9 = '0009'
    w9, b9, cost9 = Client(ID9, training_data9, test_data, sizes, _weights, _biases)
    w9 = np.array(w9, dtype=object)
    b9 = np.array(b9, dtype=object)
    ID10 = '0010'
    w10, b10, cost10 = Client(ID10, training_data10, test_data, sizes, _weights, _biases)
    w10 = np.array(w10, dtype=object)
    b10 = np.array(b10, dtype=object)

    print("Epoch %d: accuracy rate: %.2f%%" % (E, evaluate(test_data,_biases,_weights) / n_test * 100))
    #输出最后的损失函数
    print("cost:",cost1)
    weights=np.array(_weights, dtype=object)
    biases=np.array(_biases, dtype=object)
    E += 1