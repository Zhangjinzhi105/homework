import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


# 加载数据集
def load_datasets(is_plot=True):
    np.random.seed(3)

    # 生成数据集
    train_x, train_y = sklearn.datasets.make_moons(300, noise=0.2)

    # 绘图
    if is_plot:
        plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    # 数据预处理
    train_x = train_x.T
    train_y = train_y.reshape(1, len(train_y))

    return train_x, train_y


# 生成mini-batch数据集
def mini_batch_datasets(X, Y, batch_size, seed=0):
    # 指定随机种子
    np.random.seed(seed)
    # 随机打乱原数据集
    m_batch = X.shape[1]
    permutation = list(np.random.permutation(m_batch))
    shuffle_X = X[:, permutation]
    shuffle_Y = Y[:, permutation]

    # 重新划分mini-batch数据集
    mini_batchs = []
    # 如果m_batch / batch_size不是整除， 舍弃
    m_mini_batch = int(np.floor(m_batch / batch_size)) # 将float转换为int
    for i in range(m_mini_batch):
        mini_batch_X = shuffle_X[:, i * batch_size : (i + 1) * batch_size]
        mini_batch_Y = shuffle_Y[:, i * batch_size : (i + 1) * batch_size]
        mini_batch = [mini_batch_X, mini_batch_Y]
        mini_batchs.append(mini_batch)

    # 如果m_batch / batch_size不是整除，处理余下数据
    if m_batch // batch_size != 0:
        mini_batch_X = shuffle_X[:, batch_size * m_mini_batch :]
        mini_batch_Y = shuffle_Y[:, batch_size * m_mini_batch :]
        mini_batch = [mini_batch_X, mini_batch_Y]
        mini_batchs.append(mini_batch)

    return mini_batchs


# momentum算法初始化：初始化v
def init_with_momentum(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(1, L+1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v


# momentum梯度下降算法
def update_parameters_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        # dw和db的指数加权平均数
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]

        # 更新参数
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

    return parameters


# adam算法初始化:初始化s, v
def init_with_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L+1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return s, v


# adam梯度下降算法
def update_parameters_adam(parameters, grads, v, s, beta1, beta2, learning_rate, epsilon, t):
    L = len(parameters) // 2
    v_corrected = {} # 偏差修正后的值
    s_corrected = {} # 偏差修正后的值
    for l in range(1, L + 1):
        # dw和db的指数加权平均数
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        # 指数加权平均数的偏差修正
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))

        # dw**2和db**2的指数加权平均数
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])
        # dw**2和db**2的偏差修正
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))

        # 更新参数
        parameters["W" + str(l)] = parameters["W" + str(l)] \
                                   - learning_rate * v_corrected["dW" + str(l)] \
                                   / np.sqrt(s_corrected["dW" + str(l)] + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] \
                                   - learning_rate * v_corrected["db" + str(l)] \
                                   / np.sqrt(s_corrected["db" + str(l)] + epsilon)

    return parameters


# 初始化参数
def init_parameters(layers_dims):
    np.random.seed(3)

    # 获取网络层数（包含输入层）
    L = len(layers_dims)
    parameters = {}

    # 随机初始化参数
    for l in range(1, L):
        parameters["W" + str(l)]= np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


# sigmoid函数
def sigmoid(z):
    A = 1 / (1 + np.exp(- z))
    cache = z

    return A, cache


# relu函数
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


# 线性函数
def liner_forward(W, b, A_prev):
    z = np.dot(W, A_prev) + b
    cache = (W, b, A_prev)

    return z, cache


# 线性激活函数
def liner_activation_forward(W, b, A_prev, activation):
    if activation == "sigmoid":
        z, liner_cache = liner_forward(W, b, A_prev)
        A, liner_action_cache = sigmoid(z)
    elif activation == "relu":
        z, liner_cache = liner_forward(W, b, A_prev)
        A, liner_action_cache = relu(z)

    cache = (liner_cache, liner_action_cache)

    return A, cache


# 前向传播
def L_model_forward(parameters, X):
    L = len(parameters) // 2
    A_prev = X
    caches = []
    for l in range(1, L):
        A_prev, cache_l = liner_activation_forward(parameters["W" + str(l)], parameters["b" + str(l)] , A_prev, "relu")
        caches.append(cache_l)

    AL, cache_L = liner_activation_forward(parameters["W" + str(L)], parameters["b" + str(L)] , A_prev, "sigmoid")
    caches.append(cache_L)

    return AL, caches


# 代价函数
def compute_cost(A, Y):
    m = Y.shape[1]
    cost = (-1 / m ) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    # cost = (-1 / m ) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))

    return cost


# sigmoid函数的导数
def sigmoid_backward(dA, liner_activation_cache):
    Z = liner_activation_cache
    A = 1 / (1 + np.exp(Z))
    dZ = dA * A * (1 - A)

    return dZ


# relu函数的导数
def relu_backward(dA, liner_activation_cache):
    Z = liner_activation_cache
    dZ = dA
    dZ[Z <= 0] = 0

    return dZ


# 线性函数的导数
def liner_backward(dZ, liner_cache):
    W = liner_cache[0]
    A_prev = liner_cache[2]
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dW, db, dA_prev


# 线性激活函数的导数
def liner_activation_backward(dA, cache, activation):
    liner_cache = cache[0]
    liner_activation_cache = cache[1]

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, liner_activation_cache)
        dW, db, dA_prev = liner_backward(dZ, liner_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, liner_activation_cache)
        dW, db, dA_prev = liner_backward(dZ, liner_cache)

    return dW, db, dA_prev


# 反向传播
def L_model_backward(A, Y, parameters, caches):
    dAL = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    m = Y.shape[1]
    L = len(parameters) // 2
    cache = caches[L - 1]
    grads = {}

    dW, db, dA_prev = liner_activation_backward(dAL, cache, "sigmoid")
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in reversed(range(1, L)):
        cache = caches[l - 1]
        dW, db, dA_prev = liner_activation_backward(dA_prev, cache, "relu")

        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db

    return  grads


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] =  parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


# 预测方法
def predict(X, Y, parameters):
    A, cache = L_model_forward(parameters, X)
    m = A.shape[1]

    for i in range(m):
       A[0][i] = 1 if A[0][i] > 0.5 else 0

    print("准确度为: " + str(float(np.sum((A == Y)) / m)))


# 启动模型
def model(X, Y, learning_rate, layers_dims, iteration_nums, optimizer, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # 随机种子
    seed = 10
    # 初始化参数
    parameters = init_parameters(layers_dims)
    costs = []
    t = 0

    if optimizer == "mini_batch":
        pass
    elif optimizer == "momentum":
        v = init_with_momentum(parameters)
    elif optimizer == "adam":
        s, v = init_with_adam(parameters)
    else:
        print("optimizer参数传入错误，程序退出")
        exit(0) # 程序正常退出，默认值是0

    for i in range(iteration_nums):
        seed = seed + 1
        # 生成mini-batch数据集
        mini_batchs = mini_batch_datasets(X, Y, 64, seed)
        for mini_batch in mini_batchs:
            t = t + 1
            (minibatch_X, minibatch_Y) = mini_batch
            # 前向传播
            A, caches = L_model_forward(parameters, minibatch_X)
            # 计算代价函数
            cost = compute_cost(A, minibatch_Y)
            # 反向传播
            grads = L_model_backward(A, minibatch_Y, parameters, caches)
            # 更新参数
            if optimizer == "mini_batch":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters = update_parameters_momentum(parameters, grads, v, beta1, learning_rate)
            elif optimizer == "adam":
                parameters = update_parameters_adam(parameters, grads, v, s,
                                                    beta1, beta2, learning_rate, epsilon, t)
            else:
                print("optimizer参数传入错误，程序退出")
                exit(1) # 程序退出：程序发生了错误


        if(i % 100 == 0):
                costs.append(cost)
        if(i % 1000 == 0):
                print("第{}次迭代，cost的值为{}".format(i, cost))

    #绘制成本图
    plt.plot(costs)
    plt.title("learning_rate = {}".format(learning_rate))
    plt.xlabel("iterations(per hundred)")
    plt.ylabel("cost")
    plt.show()

    return parameters


# 加载数据
train_x, train_y = load_datasets()
# 模型架构
layers_dims = [train_x.shape[0], 5, 2, 1] #  4-layer model
# 启动模型
parameters = model(train_x, train_y, 0.0007, layers_dims, 10000, "adam")
# 预测的准确性
predictions_train = predict(train_x, train_y, parameters) #训练集





