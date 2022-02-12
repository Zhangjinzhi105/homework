import numpy as np
import scipy.io as sio
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt


# 使用sklearn.datasets生成数据集
def loaddatas_circles(is_plot=True):
    # 生成数据集
    np.random.seed(1)
    train_x, train_y = sklearn.datasets.make_circles(300, noise = 0.05)
    np.random.seed(2)
    test_x, test_y = sklearn.datasets.make_circles(100, noise=0.05)

    # 可视化数据集
    if is_plot:
        plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=plt.cm.Spectral)
        plt.show()

    # 预处理数据集
    train_x = train_x.T
    train_y = train_y.reshape(1, len(train_y))
    test_x = test_x.T
    test_y = test_y.reshape(1, len(test_y))

    return train_x, train_y, test_x, test_y


# 将权重参数初始化为0
def init_with_zero(layers_dim):
    L = len(layers_dim)
    parameters = {}
    for l in range(1, L):
        parameters["w" + str(l)] = np.zeros((layers_dim[l], layers_dim[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return parameters


# 随机初始化权重参数
def init_with_rand(layers_dim):
    np.random.seed(3)  # 指定随机种子
    L = len(layers_dim)
    parameters = {}
    for l in range(1, L):
        parameters["w" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return parameters


# 合适的随机初始化权重参数
def init_with_sqrt(layers_dim):
    np.random.seed(3)  # 指定随机种子
    L = len(layers_dim)
    parameters = {}
    for l in range(1, L):
        # parameters["w" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * np.sqrt(2 / layers_dim[l - 1])
        parameters["w" + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1]) * np.sqrt(1 / layers_dim[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dim[l], 1))

    return parameters


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播
def forward_propagate(train_x, parameters):
    # 获取权重参数
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]

    # 三层网络的前向传播
    z1 = np.dot(w1, train_x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)

    cache = z1, a1, w1, z2, a2, w2, z3, a3, w3

    return a3, cache


# 计算代价函数
def compute_cost(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    return cost


# 反向传播
def backward_propagate(X, Y, cache):
    z1, a1, w1, z2, a2,w2,  z3, a3, w3 = cache
    m = Y.shape[1]

    dz3 = a3 - Y
    dW3 = (1 / m) * np.dot(dz3, a2.T)
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(w3.T, dz3)
    dz2 = da2
    dz2[z2 < 0] = 0
    dW2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    dz1 = da1
    dz1[z1 < 0] = 0
    dW1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        "dW1" : dW1,
        "db1" : db1,
        "dW2" : dW2,
        "db2" : db2,
        "dW3" : dW3,
        "db3" : db3
    }

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1,L+1):
        parameters["w" + str(l)] = parameters["w" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


# 预测
def predict(X, Y, parameters):
    A, cache = forward_propagate(X, parameters)
    m = Y.shape[1]
    for i in range(A.shape[1]):
        A[0, i] = 1 if  A[0, i] > 0.5 else 0

    print("准确度为: " + str(float(np.sum((A == Y)) / m)))

    return A, cache


# 用3种不同的初始化方式启动模型
def model1(X, Y, iteration_num, learning_rate, init, layers_dim, is_plot=True):
    # np.set_printoptions(precision=11)
    # 初始化参数
    if init == "zero":
        parameters = init_with_zero(layers_dim)
    elif init == "rand":
        parameters = init_with_rand(layers_dim)
    elif init == "sqrt":
        parameters = init_with_sqrt(layers_dim)
    else:
        print("初始化参数错误！")
        exit

    costs = []

    # 多次迭代实现梯度下降
    for i in range(iteration_num):
        # 前向传播
        A, cache = forward_propagate(X, parameters)
        # 代价函数
        cost = compute_cost(A, Y)
        costs.append(cost)
        # 后向传播
        grads = backward_propagate(X, Y, cache)
        # grads = init_utils.backward_propagation(X, Y, cache)
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        if i != 0 and i % 1000 == 0:
            print("第" + str(i) + "次迭代，cost的值为：" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.title("learning_rate = {}".format(learning_rate))
        plt.xlabel("iterations(per thousand)")
        plt.ylabel(cost)
        plt.show()

    return parameters


# train_x, train_y, test_x, test_y = loaddatas_circles()
# layers_dim = [train_x.shape[0], 10, 5, 1]
# parameters = model1(train_x, train_y, 15000, 0.01, "sqrt", layers_dim)
# print("训练集准确度：")
# predict_train, cache = predict(train_x, train_y, parameters)
# print("测试集准确度：")
# predict_test, cache = predict(test_x, test_y, parameters)
# print("predict_train:" + str(predict_train))
# print("predict_test:" + str(predict_test))


# 加载.mat类型数据文件
def loaddatas_mat():
    # 加载数据文件
    data = sio.loadmat("./datasets/data.mat")
    # 获取数据集
    train_x = data["X"]
    train_y = data["y"]
    test_x = data["Xval"]
    test_y = data["yval"]
    # 数据预处理
    train_x = train_x.T
    train_y = train_y.T
    test_x = test_x.T
    test_y = test_y.T

    plt.scatter(train_x[0, :], train_x[1, :], c=train_y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    return train_x, train_y, test_x, test_y


# l2范数正则化:代价函数
def compute_cost_l2(A, Y, cache, lambd):
    z1, a1, w1, z2, a2, w2, z3, a3, w3 = cache
    m = Y.shape[1]
    cost_primary = compute_cost(A, Y)
    l2_norm = np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3))
    cost_l2 = lambd / np.multiply(2, m) * l2_norm
    cost = cost_primary + cost_l2

    return cost


# l2范数 + dropout正则化:代价函数
def compute_cost_l2_dropout(A, Y, cache, lambd):
    z1, a1, w1, b1, z2, a2, w2, b2, z3, a3, w3 = cache
    m = Y.shape[1]
    cost_primary = compute_cost(A, Y)
    l2_norm = np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3))
    cost_l2 = lambd / np.multiply(2, m) * l2_norm
    cost = cost_primary + cost_l2

    return cost


# l2范数正则化：反向传播
def backward_propagate_l2(X, Y, cache, lambd):
    z1, a1, w1, z2, a2, w2, z3, a3, w3 = cache
    m = Y.shape[1]

    dz3 = a3 - Y
    dW3 = (1 / m) * np.dot(dz3, a2.T) + lambd / m * w3
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(w3.T, dz3)
    dz2 = da2
    dz2[z2 < 0] = 0
    dW2 = (1 / m) * np.dot(dz2, a1.T) + lambd / m * w2
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    dz1 = da1
    dz1[z1 < 0] = 0
    dW1 = (1 / m) * np.dot(dz1, X.T) + lambd / m * w1
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }

    return grads


# dropout正则化：前向传播
def forward_propagate_dropout(X, parameters, keep_prob):
    np.random.seed(1)
    # 获取权重参数
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]

    # 三层网络的前向传播
    z1 = np.dot(w1, train_x) + b1
    a1 = relu(z1)
    d1 = np.random.rand(a1.shape[0], a1.shape[1])
    d1 = d1 < keep_prob
    a1 = np.multiply(a1, d1)
    a1 = a1 / keep_prob

    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    d2 = np.random.rand(a2.shape[0], a2.shape[1])
    d2 = d2 < keep_prob
    a2 = np.multiply(a2, d2)
    a2 = a2 / keep_prob

    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)

    cache = z1, a1, w1, d1, z2, a2, w2, d2, z3, a3, w3

    return a3, cache


# dropout正则化：反向传播
def backward_propagate_dropout(X, Y, cache, keep_prob):
    z1, a1, w1, d1, z2, a2, w2, d2, z3, a3, w3 = cache
    m = Y.shape[1]

    dz3 = a3 - Y
    dW3 = (1 / m) * np.dot(dz3, a2.T)
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(w3.T, dz3)
    da2 = np.multiply(da2, d2)
    da2 = da2 / keep_prob
    dz2 = da2
    dz2[z2 < 0] = 0
    dW2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    da1 = np.multiply(da1, d1)
    da1 = da1 / keep_prob
    dz1 = da1
    dz1[z1 < 0] = 0
    dW1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        "dW1" : dW1,
        "db1" : db1,
        "dW2" : dW2,
        "db2" : db2,
        "dW3" : dW3,
        "db3" : db3
    }

    return grads


# l2范数 + dropout正则化：反向传播
def backward_propagate_l2_dropout(X, Y, cache, lambd, keep_prob):
    z1, a1, w1, d1, z2, a2, w2, d2, z3, a3, w3 = cache
    m = Y.shape[1]

    dz3 = a3 - Y
    dW3 = (1 / m) * np.dot(dz3, a2.T) + lambd / m * w3
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(w3.T, dz3)
    da2 = np.multiply(da2, d2)
    da2 = da2 / keep_prob
    dz2 = da2
    dz2[z2 < 0] = 0
    dW2 = (1 / m) * np.dot(dz2, a1.T) + lambd / m * w2
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    da1 = np.multiply(da1, d1)
    da1 = da1 / keep_prob
    dz1 = da1
    dz1[z1 < 0] = 0
    dW1 = (1 / m) * np.dot(dz1, X.T) + lambd / m * w1
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }

    return grads


# 启动模型：采用l2范数正则化，dropout正则化
def model2(X, Y, learning_rate, iteration_num, layers_dim, lambd=0, keep_prob=1, is_plot=True):
    # 初始化权重参数
    parameters = init_with_sqrt(layers_dim)
    costs = []
    grads = {}
    #多次迭代实现梯度下降算法
    for i in range(iteration_num):
        # 是否使用dropout正则化
        if keep_prob == 1: # 不使用dropout
            # 前向传播
            A, cache = forward_propagate(X, parameters)
        elif keep_prob < 1: # 使用dropout
            # 前向传播
            A, cache = forward_propagate_dropout(X, parameters, keep_prob)
        else:
            print("dropout参数传入错误！")
            exit

        # 是否使用l2范数正则化
        if lambd == 0: # 不使用l2范数正则化
            # 代价函数
            cost = compute_cost(A, Y)
        elif lambd > 0 and keep_prob < 1:  # 同时使用l2范数正则化和dropout正则化
            # 代价函数
            cost = compute_cost_l2_dropout(A, Y, cache, lambd)
        elif lambd > 0: # 仅使用l2范数正则化
            # 代价函数
            cost = compute_cost_l2(A, Y, cache, lambd)
        else:
            print("lambd参数传入错误！")
            exit

        if lambd == 0 and keep_prob == 1: # 不使用任何正则化
            # 反向传播
            grads = backward_propagate(X, Y, cache)
        elif lambd > 0 and keep_prob < 1: # 同时使用l2范数正则化和dropout正则化
            # 反向传播
            grads = backward_propagate_l2_dropout(X, Y, cache, lambd, keep_prob)
        elif lambd > 0: # 使用l2范数正则化
            # 反向传播
            grads = backward_propagate_l2(X, Y, cache, lambd)
        elif keep_prob < 1: # 使用dropout正则化
            # 反向传播
            grads = backward_propagate_dropout(X, Y, cache, keep_prob)
        else:
            print("lambd参数或者dropout参数传入错误！")
            exit


        #更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
        if i % 10000 == 0:
            print("第" + str(i) + "次迭代，cost的值为：" + str(cost))

    # 绘制损失函数与迭代次数的2D图
    if is_plot:
        plt.plot(costs)
        plt.title("learning_rate = {} ".format(learning_rate))
        plt.xlabel("iterations(per thousand)")
        plt.ylabel(cost)
        plt.show()

    return parameters, grads


train_x, train_y, test_x, test_y = loaddatas_mat()
layers_dims = [train_x.shape[0], 20, 3, 1]
# parameters, grads = model2(train_x, train_y, 0.3, 30000, layers_dims, lambd=0.7, keep_prob=0.86)
parameters, grads = model2(train_x, train_y, 0.3, 30000, layers_dims, lambd=0.7)
print("训练集：")
predict_train, cache = predict(train_x, train_y, parameters)
print("测试集：")
predict_test, cache = predict(test_x, test_y, parameters)


# 将权重参数字典转换为向量
def dic_to_vector_parameters(parameters):
    L = len(parameters) // 2
    # 将字典转换为向量
    theta = []
    for l in range(1, L+1):
        vector_new_w = parameters["w" + str(l)].reshape(-1, 1)
        vector_new_b = parameters["b" + str(l)].reshape(-1, 1)

        if len(theta) == 0:
            theta = np.concatenate((vector_new_w, vector_new_b))
        elif len(theta) > 0:
            theta = np.concatenate((theta, vector_new_w, vector_new_b))

    return theta


# 将梯度参数字典转换为向量
def dic_to_vector_grads(grads):
    L = len(grads) // 2
    # 将字典转换为向量
    grad = []
    for l in range(1, L + 1):
        vector_new_dW = grads["dW" + str(l)].reshape(-1, 1)
        vector_new_db = grads["db" + str(l)].reshape(-1, 1)

        if len(grad) == 0:
            grad = np.concatenate((vector_new_dW, vector_new_db))
        elif len(grad) > 0:
            grad = np.concatenate((grad, vector_new_dW, vector_new_db))

    return grad


# 将theta转换vector
def vector_to_parameters(theta):
    w1 = theta[:40].reshape(20, 2)
    b1 = theta[40:60].reshape(20, 1)
    w2 = theta[60:120].reshape(3, 20)
    b2 = theta[120:123].reshape(3, 1)
    w3 = theta[123:126].reshape(1, 3)
    b3 = theta[126:].reshape(1, 1)

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3
    }

    return parameters


# 梯度校验:使用l2范数
def gard_check(X, Y, parameters, grads, epsilon=1e-7):
    np.set_printoptions(precision=16)
    theta = dic_to_vector_parameters(parameters)
    grad = dic_to_vector_grads(grads)
    num = len(theta)
    dtheta = np.zeros((num, 1))

    # 利用双边公差计算dtheta
    for i in range(num):
        # 计算J(θ + ε)
        theta_plus = np.copy(theta)
        theta_plus[i] = theta_plus[i] + epsilon
        parameters = vector_to_parameters(theta_plus)
        A_plus, cache = forward_propagate(X, parameters)
        J_theta_plus = compute_cost_l2(A_plus, Y, cache, lambd=0.7)

        # 计算J(θ - ε)
        theta_minus = np.copy(theta)
        theta_minus[i] = theta_minus[i] - epsilon
        parameters = vector_to_parameters(theta_minus)
        A_minus, cache = forward_propagate(X, parameters)
        J_theta_minus = compute_cost_l2(A_minus, Y, cache, lambd=0.7)

        # 计算d(θ[i])
        dtheta[i] = (J_theta_plus - J_theta_minus) / np.multiply(2, epsilon)

    # 计算梯度检验误差
    norm_numerator = np.linalg.norm(dtheta - grad)
    norm_denominator = np.linalg.norm(dtheta) +  np.linalg.norm(grad)
    difference = norm_numerator / norm_denominator

    if difference < 1e-7:
        print("梯度检验：梯度正常！")
    else:
        print("梯度检验：梯度超出阈值！")
    print("梯度检验的误差为：{}".format(difference))


gard_check(train_x, train_y, parameters, grads)
