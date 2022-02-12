import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset

# 加载数据集
X, Y = load_planar_dataset()


# 构建神经网络架构
def layer_sizes(X, Y, hide_layers_num):
    # 输入层隐藏单元个数
    input_layers_num = X.shape[0]
    # 隐藏层隐藏单元个数
    hide_layers_num = hide_layers_num
    # 输出层隐藏单元个数
    output_layers_num = Y.shape[0]

    return input_layers_num, hide_layers_num, output_layers_num


# 初始化参数（2层神经网络）
def init_parameter(input_layers_num, hide_layers_num, output_layers_num):
    W1 = np.random.randn(hide_layers_num, input_layers_num) * 0.01
    b1 = np.zeros((hide_layers_num, 1))
    W2 = np.random.randn(output_layers_num, hide_layers_num) * 0.01
    b2 = np.zeros((output_layers_num, 1))

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }

    return parameters


# sigmoid()函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# tanh()函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# 正向传播
def front_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return cache


# 计算成本函数
def compute_cost(cache, Y):
    A2 = cache["A2"]
    m = Y.shape[1]

    cost = (-1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * (np.log(1 - A2)))

    return cost


# 计算反向传播
def back_propagatation(parameters, cache, Y):
    A2 = cache["A2"]
    A1 = cache["A1"]
    W2 = parameters["W2"]
    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW2" : dW2,
        "db2" : db2,
        "dW1" : dW1,
        "db1" : db1
    }

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# 预测方法
def predict(X, parameters):
    cache = front_propagation(X, parameters)
    predictions = cache["A2"]

    return np.round(predictions)


# 启动模型
def model(X, Y, iteration_num, learning_rate, hide_layers_num):
    # 构建神经网络架构
    input_layers_num, hide_layers_num, output_layers_num = layer_sizes(X, Y, hide_layers_num)
    # 初始化参数
    parameters = init_parameter(input_layers_num, hide_layers_num, output_layers_num)
    # 循环迭代梯度下降法
    for i in range(iteration_num):
        # 计算前向传播
        cache = front_propagation(X, parameters)
        # 计算成本函数
        cost = compute_cost(cache, Y)
        # 计算反向传播
        grads = back_propagatation(parameters, cache, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if(i % 1000 ==0):
            print("第{}次梯度下降：cost的值是{}".format(i, cost))

    return parameters

parameters = model(X, Y, 10000, 0.5, 4)
predictions = predict(X, parameters)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
