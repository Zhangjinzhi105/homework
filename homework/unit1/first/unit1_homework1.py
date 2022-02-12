import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

# 加载数据：训练集数据和测试集数据
def load_dataset():
    train_location = "./datasets/train_catvnoncat.h5"
    test_location = "./datasets/test_catvnoncat.h5"

    # 获取训练集数据
    train_file = h5py.File(train_location, "r")
    train_set_x_original = np.array(train_file["train_set_x"][:])
    train_set_y_original = np.array(train_file["train_set_y"][:])

    # 获取测试集数据
    test_file = h5py.File(test_location, "r")
    test_set_x_original = np.array(test_file["test_set_x"][:])
    test_set_y_original = np.array(test_file["test_set_y"][:])

    #重 新设定训练集和测试集矩阵的维度:X = n*m   Y = 1*m
    train_set_x = train_set_x_original.reshape(train_set_x_original.shape[0], -1).T
    train_set_y = train_set_y_original.reshape(1, train_set_y_original.shape[0])
    test_set_x = test_set_x_original.reshape(test_set_x_original.shape[0], -1).T
    test_set_y = test_set_y_original.reshape(1, test_set_y_original.shape[0])

    train_set_x = train_set_x / 255
    test_set_x = test_set_x / 255

    return train_set_x, train_set_y, test_set_x, test_set_y


# 初始化w和b的权重值
def init_weights(dim):
    W = np.zeros((dim, 1))
    b = 0

    # 定义矩阵时，使用assert确保其维度
    assert(W.shape == (dim, 1))
    # 定义变量时，使用assert确保其类型
    assert(isinstance(b, int) or isinstance(b, float))

    return W, b


# 定义激活函数
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# 正向和反向传播
def propagate(W, b, X, Y):
    m = X.shape[1]

    # 正向传播
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)

    # 反向传播
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) # 计算成本
    cost = np.round(cost, 6) # 取小数点后6位，四舍五入
    dZ = A - Y
    dW = (1 / m) * np.dot(X, dZ.T)
    db = (1 / m) * np.sum(dZ)

    # 创建一个字典，把dW, db的值保存起来
    grads = {
        "dW" : dW,
        "db" : db
    }
    return grads, cost


# 优化梯度下降算法
def optimize(W, b, X, Y, learning_rate, num_iterations):
    costs = []
    for i in range(num_iterations):
        #正向和反向传播
        grads, cost = propagate(W, b, X, Y)
        dW = grads["dW"]
        db = grads["db"]

        #更新权重值
        W = W - learning_rate * dW
        b = b - learning_rate * db

        #每循环100次，打印输出一次代价函数的值
        if i % 100 == 0:
            print("第{}次循环,代价函数的值为{}".format(i, cost))
            costs.append(cost)

        #创建一个字典，保存参数
        params = {
            "W" : W,
            "b" : b
        }
        grads = {
            "dW": dW,
            "db": db
        }

    return params, grads, costs


# 预测方法
def predict(W, b, X):
    # 计算预测值A
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)

    # 定义矩阵Y
    Y = np.zeros((1, X.shape[1]))
    for i in range(X.shape[1]):
        Y[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y

# 启动模型
def model():
    # 加载数据
    train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()
    # 初始化权重
    W, b = init_weights(train_set_x.shape[0])
    # 梯度下降算法
    params, grads,costs = optimize(W, b, train_set_x, train_set_y, 0.005, 2000)

    # 计算训练集准确率
    train_predict_Y = predict(params["W"], params["b"], train_set_x)
    train_accuracy_rate = 100 - 100 * np.mean(np.abs(train_predict_Y - train_set_y))
    print("训练集准确性：{} %".format(train_accuracy_rate))

    # 计算测试集准确率
    test_predict_Y = predict(params["W"], params["b"], test_set_x)
    test_accuracy_rate = 100 - 100 * np.mean(np.abs(test_predict_Y - test_set_y))
    print("训练集准确性：{} %".format(test_accuracy_rate))


model()
