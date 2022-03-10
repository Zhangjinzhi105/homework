import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

"""
如果你有一个Tensro t ，在使用t.eval()时，等价于：tf.get_default_session().run(t)

run(）可以同时获取多个Tensor中的值；eval()只能一次获取一个Tensor值。

eval()：只能用于tf.Tensor类对象，也就是有输出的Operation。对于没有输出的Operation，可以用run（）.

每次使用eval()和run()时，都会执行整个计算图。
"""


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
    train_x = train_x_orginal.reshape(train_x_orginal.shape[0], -1).T
    test_x = test_x_orginal.reshape(test_x_orginal.shape[0], -1).T
    # 归一化数据
    train_x = train_x / 255
    test_x = test_x / 255
    # 统一维度
    train_y = train_y_orginal.reshape(1, train_y_orginal.shape[0])
    test_y = test_y_orginal.reshape(1, test_y_orginal.shape[0])
    # 转为独热编码
    train_y = tf.transpose(tf.one_hot(np.squeeze(train_y), 6))
    test_y = tf.transpose(tf.one_hot(np.squeeze(test_y), 6))
    # np.eye()的函数，除了生成对角阵外，还可以将一个label数组，大小为(1,m)或者(m,1)的数组，转化成one-hot数组。
    # train_y = np.eye(6)[train_y.reshape(-1)].T
    # test_y = np.eye(6)[test_y.reshape(-1)].T

    return train_x, train_y, test_x, test_y


# 定义占位符
def create_placeholder(n_x, n_y):
    X = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y


# 初始化参数
def init_parameters():
    tf.compat.v1.set_random_seed(1)  # 指定随机种子

    W1 = tf.compat.v1.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.compat.v1.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.compat.v1.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.compat.v1.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
    }

    return parameters


# 前向传播
def forward_propagate(X, parameters):
    # 获取参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # 前向传播
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    return Z3


# 计算代价函数
def compute_cost(Z3, Y):
    Z3_transpose = tf.transpose(Z3)
    Y_transpose = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_transpose, logits=Z3_transpose))

    return cost


# 处理mini_batch:确保传入的X, Y都是ndarray类型，不是Tensor类型
def random_mini_batches(X, Y, minibatch_size, seed):
    # 打乱顺序
    m = X.shape[1]
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    X_shuffle = X[:, permutation]
    Y_shuffle = Y[:, permutation]

    # 分批处理
    minibatch_num = m // minibatch_size
    mini_batches = []
    for i in range(minibatch_num):
        minibatch_X = X_shuffle[:, i * minibatch_size :(i + 1) * minibatch_size]
        minibatch_Y = Y_shuffle[:, i * minibatch_size :(i + 1) * minibatch_size]
        mini_batch = [minibatch_X, minibatch_Y]
        mini_batches.append(mini_batch)
    # 若没有整除
    if m % minibatch_size != 0:
        minibatch_X = X_shuffle[:, minibatch_num * minibatch_size:]
        minibatch_Y = Y_shuffle[:, minibatch_num * minibatch_size:]
        mini_batch = [minibatch_X, minibatch_Y]
        mini_batches.append(mini_batch)

    return mini_batches


# 启动模型
def model(train_X, train_Y, test_X, test_Y, learning_rate, epoch_num, minibatch_size, is_plot=True):
    # 定义占位符
    (n_x, m) = train_X.shape # int类型
    n_y = train_Y.shape[0]
    X, Y = create_placeholder(n_x, n_y)
    # 初始化参数
    parameters = init_parameters()
    # 前向传播
    Z3 = forward_propagate(X, parameters)
    # 代价函数
    cost = compute_cost(Z3, Y)
    # 反向传播&更新参数
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 初始化变量:只有使用Variable()时，使用梯度下降时，才使用init
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as session:
        # 初始化
        session.run(init)
        seed = 3
        costs = []

        # 循环迭代
        for i in range(epoch_num):
            epoch_cost = 0
            seed = seed + 1
            minibatch_num = m // minibatch_size # "/ "表示浮点数除法，返回浮点结果;"//"表示整数除法,返回不大于结果的一个最大的整数
            mini_batches = random_mini_batches(train_X, train_Y.eval(), minibatch_size, seed)
            for mini_batch in mini_batches:
                mini_batch_X, mini_batch_Y = mini_batch
                _, minibatch_cost = session.run([optimizer, cost], feed_dict=({X : mini_batch_X, Y : mini_batch_Y}))
                epoch_cost = epoch_cost + minibatch_cost / minibatch_num
            # 缓存cost
            if i % 5 == 0:
                costs.append(epoch_cost)
            # 打印cost
            if i % 100 == 0:
                print("第{}次迭代，cost的值是{}".format(i, epoch_cost))

        # 绘制成本图
        if is_plot:
            plt.plot(costs)
            plt.xlabel("epoch_num(per five)")
            plt.ylabel("cost")
            plt.title("learning_rate: " + str(learning_rate))
            plt.show()

        # 将parmeters保存进session
        parameters = session.run(parameters)

        # 预测的准确率
        prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
        # print("训练集准确率：" + str(np.squeeze(accuracy.eval({X : train_X, Y : train_Y.eval()}))))
        # print("测试集准确率：" + str(np.squeeze(accuracy.eval({X : test_X, Y : test_Y.eval()}))))
        print("训练集准确率：" + str(np.squeeze(session.run(accuracy, feed_dict={X: train_X, Y: train_Y.eval()}))))
        print("测试集准确率：" + str(np.squeeze(session.run(accuracy, feed_dict={X: test_X, Y: test_Y.eval()}))))


    return parameters


# 加载数据
train_x, train_y, test_x, test_y = loaddata()
# 记录开始时间
start_time = time.clock()
# 启动模型
parameters = model(train_x, train_y, test_x, test_y, 0.0001, 1500, 32)
# 记录结束时间
end_time = time.clock()
# 打印CPU运行时间
print("CPU运行时间：" + str(end_time - start_time))


# 预测方法
def predict(X, parameters):
    x = tf.placeholder(tf.float32, shape=[12288, 1])
    Z3 = forward_propagate(X, parameters)
    p = tf.argmax(Z3)
    with tf.compat.v1.Session() as session:
        prediction = session.run(p, feed_dict={x : X})

    return prediction


# 测试自己图片
def test():
    for i in range(1, 6):
        # 获取图片地址
        image = str(i) + ".png"
        image_file = "images/fingers/" + image
        # 读取图片
        image_read = mpimg.imread((image_file))
        # 展示图片
        plt.imshow(image_read)
        # 预处理数据
        image_X = image_read.reshape(1, 64 * 64 * 3).T
        # 预测
        p = predict(image_X, parameters)
        print("第{}张图片是： {} ".format(i, p))


test()
