import numpy as np
import h5py
import matplotlib.pyplot as plt


# 加载数据
def load_datasets():
    # 文件的地址
    train_file_path = "datasets/train_catvnoncat.h5"
    test_file_path = "datasets/test_catvnoncat.h5"

    # 加载文件
    train_file = h5py.File(train_file_path, "r")
    test_file = h5py.File(test_file_path, "r")

    # 获取数据
    train_dataset_x_orignal = np.array(train_file["train_set_x"][:])
    train_dataset_y_orignal = np.array(train_file["train_set_y"][:])
    test_dataset_x_orignal = np.array(test_file["test_set_x"][:])
    test_dataset_y_orignal = np.array(test_file["test_set_y"][:])
    classes = np.array(test_file["list_classes"][:])

    # 处理数据
    train_dataset_x = train_dataset_x_orignal.reshape(train_dataset_x_orignal.shape[0], -1).T
    train_dataset_y = train_dataset_y_orignal.reshape(1, train_dataset_y_orignal.shape[0])
    test_dataset_x = test_dataset_x_orignal.reshape(test_dataset_x_orignal.shape[0], -1).T
    test_dataset_y = test_dataset_y_orignal.reshape(1, test_dataset_y_orignal.shape[0])

    # 归一化数据
    train_dataset_x = train_dataset_x / 255
    test_dataset_x = test_dataset_x / 255

    return train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y, classes


# 初始化参数
def init_parameters(layers_dims):
    np.random.seed(3)
    # 获取网络层数（包含输入层）
    L = len(layers_dims)
    parameters = {}

    # 随机初始化参数
    for l in range(1, L):
        parameters["W" + str(l)]= np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
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


# 启动模型
def L_layer_model(X, Y, learning_rate, layers_dims, iteration_nums):
    np.random.seed(1)
    # 初始化参数
    parameters = init_parameters(layers_dims)
    costs = []
    for i in range(iteration_nums):
        # 前向传播
        A, caches = L_model_forward(parameters, X)
        # 计算代价函数
        cost = compute_cost(A, Y)
        # 反向传播
        grads = L_model_backward(A, Y, parameters, caches)
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        if(i % 100 == 0):
            costs.append(cost)
            print("第{}次迭代，cost的值为{}".format(i,cost))

    #绘制成本图
    plt.plot(costs)
    plt.title("learning_rate = {}".format(learning_rate))
    plt.xlabel("iterations(per hundred)")
    plt.ylabel("cost")
    plt.show()

    return parameters


# 预测方法
def predict(X, Y, parameters):
    A, cache = L_model_forward(parameters, X)
    m = A.shape[1]

    for i in range(m):
       A[0][i] = 1 if A[0][i] > 0.5 else 0

    print("准确度为: " + str(float(np.sum((A == Y)) / m)))


train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y, classes = load_datasets()
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_dataset_x, train_dataset_y, 0.0075, layers_dims, 2500)

# 预测的准确性
predictions_train = predict(train_dataset_x, train_dataset_y, parameters) #训练集
predictions_test = predict(test_dataset_x, test_dataset_y, parameters) #测试集


# 查看预测值和实际值不一样的图片
def show_misLabeled_images(X, Y, parameters):
   # 计算预测值
   A, caches = L_model_forward(parameters, X)
   m = A.shape[1]
   for i in range(m):
       A[0][i] = 1 if A[0][i] > 0.5 else 0

   # 对预测值和标签求和
   P = A + Y
   # 过滤出预测值和标签值不一样的图片
   locations = np.asarray(np.where(P == 1))
   # 获取相异图片的元素在训练集中的位置
   m_locations = locations[1]

   # 展示相异图片
   for i, m_loc in enumerate(m_locations):
       img = X[:, m_loc].reshape(64, 64 ,3)
       plt.subplot(1, 15, i + 1)
       plt.title(np.squeeze(Y)[m_loc])
       plt.axis("off")
       plt.imshow(img)

   plt.show()


show_misLabeled_images(test_dataset_x, test_dataset_y, parameters)


