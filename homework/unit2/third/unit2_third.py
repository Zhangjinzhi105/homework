import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片


# #%matplotlib inline #如果你使用的是jupyter notebook取消注释
# np.random.seed(1)
#
# y_hat = tf.constant(36,name="y_hat")            #定义y_hat为固定值36
# y = tf.constant(39,name="y")                    #定义y为固定值39
#
# loss = tf.Variable((y-y_hat)**2,name="loss" )   #为损失函数创建一个变量
#
# init = tf.global_variables_initializer()        #运行之后的初始化(ession.run(init))
#                                                 #损失变量将被初始化并准备计算
# with tf.Session() as session:                   #创建一个session并打印输出
#     session.run(init)                           #初始化变量
#     print(session.run(loss))                    #打印损失值


# 线性函数
def linear_function():
    """
    实现一个线性功能：
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    返回：
        result - 运行了session后的结果，运行的是Y = WX + b

    """

    np.random.seed(1)  # 指定随机种子

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul是矩阵乘法
    # Y = tf.matmul(W,X) + b #也可以以写成这样子

    # 创建一个session并运行它
    sess = tf.Session()
    result = sess.run(Y)

    # session使用完毕，关闭它
    sess.close()

    return result


# sigmoid函数
def sigmoid(z):
    """
    实现使用sigmoid函数计算z

    参数：
        z - 输入的值，标量或矢量

    返回：
        result - 用sigmoid计算z的值

    """

    # 创建一个占位符x，名字叫“x”
    x = tf.placeholder(tf.float32, name="x")

    # 计算sigmoid(z)
    sigmoid = tf.sigmoid(x)

    # 创建一个会话，使用方法二
    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result


# 独热编码
def one_hot_matrix(lables, C):
    """
    创建一个矩阵，其中第i行对应第i个类号，第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签，那么entry (i,j)将会是1

    参数：
        lables - 标签向量
        C - 分类数

    返回：
        one_hot - 独热矩阵

    """

    # 创建一个tf.constant，赋值为C，名字叫C
    C = tf.constant(C, name="C")

    # 使用tf.one_hot，注意一下axis
    one_hot_matrix = tf.one_hot(indices=lables, depth=C, axis=0)

    # 创建一个session
    sess = tf.Session()

    # 运行session
    one_hot = sess.run(one_hot_matrix)

    # 关闭session
    sess.close()

    return one_hot


# 初始化
def ones(shape):
    """
    创建一个维度为shape的变量，其值全为1

    参数：
        shape - 你要创建的数组的维度

    返回：
        ones - 只包含1的数组
    """

    # 使用tf.ones()
    ones = tf.ones(shape)

    # 创建会话
    sess = tf.Session()

    # 运行会话
    ones = sess.run(ones)

    # 关闭会话
    sess.close()

    return ones

# 转换我独热编码
def convert_to_one_hot(Y, C):
    # Y = np.eye(C)[Y.reshape(-1)].T
    Y = tf.one_hot(Y, C)
    return Y


# 加载数据集
def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    X_train_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 每一列就是一个样本
    X_test_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # 归一化数据
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    # 转换为独热矩阵
    Y_train = tf_utils.convert_to_one_hot(train_set_y_orig, 6)
    Y_test = tf_utils.convert_to_one_hot(test_set_y_orig, 6)

    return X_train, Y_train, X_test, Y_test, classes


# 创建placeholder：为X和Y创建占位符，稍后在运行会话时传递训练数据。
def create_placeholders(n_x, n_y):
    """
    为TensorFlow会话创建占位符
    参数：
        n_x - 一个实数，图片向量的大小（64*64*3 = 12288）
        n_y - 一个实数，分类数（从0到5，所以n_y = 6）

    返回：
        X - 一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
        Y - 一个对应输入的标签的占位符，维度为[n_Y,None]，dtype = "float"

    提示：
        使用None，因为它让我们可以灵活处理占位符提供的样本数量。事实上，测试/训练期间的样本数量是不同的。

    """

    X = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X") # [None, 3] 表示列是3，行不定
    Y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


# 初始化权重
def initialize_parameters():
    """
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    返回：
        parameters - 包含了W和b的字典


    可使用tf.get_variable( ) 函数代替tf.Variable( )。
    如果变量存在，函数tf.get_variable( ) 会返回现有的变量。
    如果变量不存在，会根据给定形状和初始值创建变量。

    """

    tf.set_random_seed(1)  # 指定随机种子

    W1 = tf.compat.v1.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1)) # 用来保持每一层的梯度大小都差不多相同
    b1 = tf.compat.v1.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.compat.v1.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.compat.v1.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# tf.compat.v1.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。


# 前向传播
def forward_propagation(X, parameters):
    """
    实现一个模型的前向传播，模型结构为LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    参数：
        X - 输入数据的占位符，维度为（输入节点数量，样本数量）
        parameters - 包含了W和b的参数的字典

    返回：
        Z3 - 最后一个LINEAR节点的输出

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    # Z1 = tf.matmul(W1,X) + b1             #也可以这样写
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,A2) + b3

    return Z3


# 计算代价函数
def compute_cost(Z3, Y):
    """
    计算成本

    参数：
        Z3 - 前向传播的结果
        Y - 标签，一个占位符，和Z3的维度相同

    返回：
        cost - 成本值


    """
    logits = tf.transpose(Z3)  # 转置
    labels = tf.transpose(Y)  # 转置

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

# 将训练集预处理为mini_batch类型
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = np.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# 启动模型
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500,
          minibatch_size=32, print_cost=True, is_plot=True):
    """
    实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

    参数：
        X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
        Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
        X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
        Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
        learning_rate - 学习速率
        num_epochs - 整个训练集的遍历次数
        mini_batch_size - 每个小批量数据集的大小
        print_cost - 是否打印成本，每100代打印一次
        is_plot - 是否绘制曲线图

    返回：
        parameters - 学习后的参数

    """
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.compat.v1.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape  # 获取输入节点数量和样本数
    n_y = Y_train.shape[0]  # 获取输出节点数量
    costs = []  # 成本集

    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播，使用Adam优化
    # 编程框架，将所有反向传播和参数更新都在1行代码中处理。
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的变量
    init = tf.compat.v1.global_variables_initializer()

    # 开始会话并计算
    with tf.compat.v1.Session() as session:
        # 初始化
        session.run(init)

        # 正常训练的循环
        for epoch in range(num_epochs):

            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量
            seed = seed + 1
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # 选择一个minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # 数据已经准备好了，开始运行session
                # 运行tf.session时，必须将optimizer对象与成本函数一起调用，当被调用时，它将使用所选择的方法和学习速率对给定成本进行优化。
                # _ 作为一次性变量来存储稍后不需要使用的值。 这里，_具有我们不需要的优化器的评估值（并且c取值为成本变量的值）
                _, minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches # 相当于计算minibatch_cost的平均值

            # 记录并打印成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = session.run(parameters)
        print("参数已经保存到session。")

        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y)) # 返回最大的那个数值所在的下标

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


X_train, Y_train, X_test, Y_test, classes = load_dataset()
#开始时间
start_time = time.clock() # clock()函数以浮点数计算的秒数返回当前的CPU时间，用来衡量不同程序的耗时，比time.time()更有用。
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )


# 预测函数
def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation(x, params)
    p = tf.argmax(z3)

    sess = tf.compat.v1.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


# 用自己的图片测试
def test(index):
    my_image1 = str(index) + ".png"                                #定义图片名称
    fileName1 = "images/fingers/" + my_image1                      #图片地址
    image1 = mpimg.imread(fileName1)                               #读取图片
    plt.imshow(image1)                                             #显示图片
    my_image1 = image1.reshape(1,64 * 64 * 3).T                    #重构图片
    my_image_prediction = predict(my_image1, parameters)  #开始预测
    print("预测结果: y = " + str(np.squeeze(my_image_prediction)))


# 一共5张图片
for i in range(1, 6):
    test(i)