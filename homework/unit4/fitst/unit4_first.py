import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops
import cnn_utils

np.random.seed(1)  # 指定随机种子

# padding填充
def zero_pad(X, pad):
    """
    把数据集X的图像边界全部使用0来扩充pad个宽度和高度。

    参数：
        X - 图像数据集，维度为（样本数，图像高度，图像宽度，图像通道数）
        pad - 整数，每个图像在垂直和水平维度上的填充量
    返回：
        X_paded - 扩充后的图像数据集，维度为（样本数，图像高度 + 2*pad，图像宽度 + 2*pad，图像通道数）

    """

    X_paded = np.pad(X, (
        (0, 0),  # 样本数，不填充
        (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0, 0)),  # 通道数，不填充
                     'constant', constant_values=0)  # 连续一样的值填充

    return X_paded


# 单步切片卷积
def conv_single_step(a_slice_prev, W, b):
    """
    在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器。
    这里切片大小和过滤器大小相同

    参数：
        a_slice_prev - 输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
        W - 权重参数，包含在了一个矩阵中，维度为（过滤器大小，过滤器大小，上一通道数）
        b - 偏置参数，包含在了一个矩阵中，维度为（1,1,1）

    返回：
        Z - 在输入数据的片X上卷积滑动窗口（w，b）的结果。
    """

    s = np.multiply(a_slice_prev, W) + b

    Z = np.sum(s)

    return Z


# 卷积函数的前向传播
def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积函数的前向传播

    参数：
        A_prev - 上一层的激活输出矩阵，维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
        W - 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
        b - 偏置矩阵，维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
        hparameters - 包含了"stride"与 "pad"的超参数字典。

    返回：
        Z - 卷积输出，维度为(m, n_H, n_W, n_C)，（样本数，图像的高度，图像的宽度，过滤器数量）
        cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
    """

    # 获取来自上一层数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取权重矩阵的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # 计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行板除
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # 使用0来初始化卷积输出Z
    Z = np.zeros((m, n_H, n_W, n_C))

    # 通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    # 自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 执行单步卷积
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

    # 数据处理完毕，验证数据格式是否正确
    assert (Z.shape == (m, n_H, n_W, n_C))

    # 存储一些缓存值，以便于反向传播使用
    cache = (A_prev, W, b, hparameters)

    return (Z, cache)


# 池化层的前向传播
def pool_forward(A_prev, hparameters, mode="max"):
    """
    实现池化层的前向传播

    参数：
        A_prev - 输入数据，维度为(m, n_H_prev, n_W_prev, n_C_prev)
        hparameters - 包含了 "f" 和 "stride"的超参数字典
        mode - 模式选择【"max" | "average"】

    返回：
        A - 池化层的输出，维度为 (m, n_H, n_W, n_C)
        cache - 存储了一些反向传播需要用到的值，包含了输入和超参数的字典。
    """

    # 获取输入数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取超参数的信息
    f = hparameters["f"]
    stride = hparameters["stride"]

    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    # 初始化输出矩阵
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # 遍历样本
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 定位完毕，开始切割
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # 对切片进行池化操作
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_slice_prev)

    # 池化完毕，校验数据格式
    assert (A.shape == (m, n_H, n_W, n_C))

    # 校验完毕，开始存储用于反向传播的值
    cache = (A_prev, hparameters)

    return A, cache


# 加载数据
def loaddata():
    # 加载文件
    train_file = h5py.File("datasets/train_signs.h5", "r")
    test_file = h5py.File("datasets/test_signs.h5", "r")

    # 读取数据
    train_x_orginal = np.array(train_file["train_set_x"][:])
    train_y_orginal = np.array(train_file["train_set_y"][:])
    test_x_orginal = np.array(test_file["test_set_x"][:])
    test_y_orginal = np.array(test_file["test_set_y"][:])

    # 预处理数据
    train_x = train_x_orginal / 255
    test_x = test_x_orginal / 255

    # 统一维度
    train_y = train_y_orginal.reshape(1, train_y_orginal.shape[0])
    test_y = test_y_orginal.reshape(1, test_y_orginal.shape[0])

    # 转为独热编码
    train_y = tf.one_hot(np.squeeze(train_y), 6)
    test_y = tf.one_hot(np.squeeze(test_y), 6)
    # np.eye()的函数，除了生成对角阵外，还可以将一个label数组，大小为(1,m)或者(m,1)的数组，转化成one-hot数组。
    # train_y = np.eye(6)[train_y.reshape(-1)].T
    # test_y = np.eye(6)[test_y.reshape(-1)].T

    return train_x, train_y, test_x, test_y


# 创建占位符
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    为session创建占位符

    参数：
        n_H0 - 实数，输入图像的高度
        n_W0 - 实数，输入图像的宽度
        n_C0 - 实数，输入的通道数
        n_y  - 实数，分类数

    输出：
        X - 输入数据的占位符，维度为[None, n_H0, n_W0, n_C0]，类型为"float"
        Y - 输入数据的标签的占位符，维度为[None, n_y]，维度为"float"
    """
    X = tf.compat.v1.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.compat.v1.placeholder(tf.float32, [None, n_y])

    return X, Y

# 初始化
def initialize_parameters():
    """
    初始化权值矩阵，这里我们把权值矩阵硬编码：
    W1 : [4, 4, 3, 8]
    W2 : [2, 2, 8, 16]

    返回：
        包含了tensor类型的W1、W2的字典

    初始化权值/过滤器W1、W2。在这里，我们不需要考虑偏置，因为TensorFlow会考虑到的。
    需要注意的是我们只需要初始化为2D卷积函数，全连接层TensorFlow会自动初始化的。

    """
    tf.compat.v1.set_random_seed(1)

    W1 = tf.compat.v1.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.compat.v1.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# 前向传播
"""
我们具体实现的时候，我们需要使用以下的步骤和参数：
    Conv2d : 步伐：1，填充方式：“SAME”
    ReLU
    Max pool : 过滤器大小：8x8，步伐：8x8，填充方式：“SAME”
    Conv2d : 步伐：1，填充方式：“SAME”
    ReLU
    Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
    一维化上一层的输出
    全连接层（FC）：使用没有非线性激活函数的全连接层。这里不要调用SoftMax， 这将导致输出层中有6个神经元，然后再传递到softmax。 
                  在TensorFlow中，softmax和cost函数被集中到一个函数中，在计算成本时您将调用不同的函数。
"""
def forward_propagation(X, parameters):
    """
    实现前向传播
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    参数：
        X - 输入数据的placeholder，维度为(输入节点数量，样本数量)
        parameters - 包含了“W1”和“W2”的python字典。

    返回：
        Z3 - 最后一个LINEAR节点的输出

    函数简介：
    tf.nn.conv2d(X,W1,strides=[1,s,s,1],padding='SAME'):
            给定输入X和一组过滤器W1，这个函数将会自动使用W1来对X进行卷积，
            第三个输入参数是[1,s,s,1]是指对于输入 (m, n_H_prev, n_W_prev, n_C_prev)而言，每次滑动的步伐。

    tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')：
            给定输入X XX，该函数将会使用大小为（f,f）以及步伐为(s,s)的窗口对其进行滑动取最大值。

    tf.nn.relu(Z1)：计算Z1的ReLU激活

    tf.contrib.layers.flatten(P)：
            给定一个输入P，此函数将会把每个样本转化成一维的向量，然后返回一个tensor变量，其维度为（batch_size,k）.

    tf.contrib.layers.fully_connected(F, num_outputs)：
            给定一个已经一维化了的输入F，此函数将会返回一个由全连接层计算过后的输出。

    使用tf.contrib.layers.fully_connected(F, num_outputs)的时候，
    全连接层会自动初始化权值且在你训练模型的时候它也会一直参与，
    所以当我们初始化参数的时候我们不需要专门去初始化它的权值。
    我们只需要初始化为2D卷积函数，全连接层TensorFlow会自动初始化的

    """
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Conv2d : 步伐：1，填充方式：“SAME”
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    # ReLU ：
    A1 = tf.nn.relu(Z1)
    # Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
    P1 = tf.nn.max_pool2d(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    # Conv2d : 步伐：1，填充方式：“SAME”
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    # ReLU ：
    A2 = tf.nn.relu(Z2)
    # Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
    P2 = tf.nn.max_pool2d(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # 一维化上一层的输出
    P = tf.contrib.layers.flatten(P2)

    # 全连接层（FC）：使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3


# 计算代价函数
def compute_cost(Z3, Y):
    """
    计算成本
    参数：
        Z3 - 正向传播最后一个LINEAR节点的输出，维度为（6，样本数）。
        Y - 标签向量的placeholder，和Z3的维度相同

    返回：
        cost - 计算后的成本

    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

    return cost


# 处理mini_batch:确保传入的X, Y都是ndarray类型，不是Tensor类型
def random_mini_batches(X, Y, minibatch_size, seed):
    # 打乱顺序
    m = X.shape[0]
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    X_shuffle = X[permutation, :, :, :]
    Y_shuffle = Y[permutation, :]

    # 分批处理
    minibatch_num = m // minibatch_size
    mini_batches = []
    for i in range(minibatch_num):
        minibatch_X = X_shuffle[i * minibatch_size :(i + 1) * minibatch_size, :, :, :]
        minibatch_Y = Y_shuffle[i * minibatch_size :(i + 1) * minibatch_size, :]
        mini_batch = [minibatch_X, minibatch_Y]
        mini_batches.append(mini_batch)
    # 若没有整除
    if m % minibatch_size != 0:
        minibatch_X = X_shuffle[minibatch_num * minibatch_size :, :, :, :]
        minibatch_Y = Y_shuffle[minibatch_num * minibatch_size :, :]
        mini_batch = [minibatch_X, minibatch_Y]
        mini_batches.append(mini_batch)

    return mini_batches


# 启动模型
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True, isPlot=True):
    """
    使用TensorFlow实现三层的卷积神经网络
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    参数：
        X_train - 训练数据，维度为(None, 64, 64, 3)
        Y_train - 训练数据对应的标签，维度为(None, n_y = 6)
        X_test - 测试数据，维度为(None, 64, 64, 3)
        Y_test - 训练数据对应的标签，维度为(None, n_y = 6)
        learning_rate - 学习率
        num_epochs - 遍历整个数据集的次数
        minibatch_size - 每个小批量数据块的大小
        print_cost - 是否打印成本值，每遍历100次整个数据集打印一次
        isPlot - 是否绘制图谱

    返回：
        train_accuracy - 实数，训练集的准确度
        test_accuracy - 实数，测试集的准确度
        parameters - 学习后的参数
    """
    tf.compat.v1.set_random_seed(1)  # 确保你的数据和我一样
    seed = 3  # 指定numpy的随机种子
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    # 为当前维度创建占位符
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # 初始化参数
    parameters = initialize_parameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    # 计算成本
    cost = compute_cost(Z3, Y)

    # 反向传播，由于框架已经实现了反向传播，我们只需要选择一个优化器就行了
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 全局初始化所有变量
    init = tf.compat.v1.global_variables_initializer()

    # 开始运行
    with tf.compat.v1.Session() as session:
        # 初始化参数
        session.run(init)
        # 开始遍历数据集
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)  # 获取数据块的数量
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train.eval(), minibatch_size, seed)

            # 对每个数据块进行处理
            for minibatch in minibatches:
                # 选择一个数据块
                (minibatch_X, minibatch_Y) = minibatch
                # 最小化这个数据块的成本
                _, temp_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 累加数据块的成本值
                minibatch_cost += temp_cost / num_minibatches

            # 是否打印成本
            if print_cost:
                # 每5代打印一次
                if epoch % 5 == 0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

            # 记录成本
            if epoch % 1 == 0:
                costs.append(minibatch_cost)

        # 数据处理完毕，绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 将parmeters保存进session
        parameters = session.run(parameters)

        # 计算当前的预测情况
        predict_op = tf.argmax(Z3, 1)
        corrent_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        #计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
        print("corrent_prediction accuracy= " + str(accuracy))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train.eval()})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test.eval()})

        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))

    return parameters


# 预测方法
def predict(X, parameters):
    x = tf.compat.v1.placeholder(tf.float32, shape=[1, 64, 64, 3])
    Z3 = forward_propagation(X, parameters)
    p = tf.argmax(Z3, 1)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        prediction = session.run(p, feed_dict={x : X})

    return prediction


# 测试自己图片
def test(parameters):
    for i in range(1, 6):
        # 获取图片地址
        image = str(i) + ".png"
        image_file = "images/fingers/" + image
        # 读取图片
        image_read = mpimg.imread((image_file))
        # 展示图片
        plt.imshow(image_read)
        # 预处理数据
        image_X = image_read.reshape(1, 64, 64, 3)
        # 预测
        p = predict(image_X, parameters)
        print("第{}张图片是： {} ".format(i, p))

train_x, train_y, test_x, test_y = loaddata()
parameters = model(train_x, train_y, test_x, test_y, num_epochs=150)
test(parameters)

