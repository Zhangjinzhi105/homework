import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

np.random.seed(1)

# 加载数据
def load_datasets():
    # 加载文件
    train_file = h5py.File("datasets/train_signs.h5", "r")
    test_file = h5py.File("datasets/test_signs.h5", "r")

    # 获取数据
    train_x_orginal = np.array(train_file["train_set_x"][:])
    train_y_orginal = np.array(train_file["train_set_y"][:])
    test_x_orginal = np.array(test_file["test_set_x"][:])
    test_y_orginal = np.array(test_file["test_set_y"][:])

    # 预处理数据
    train_x = train_x_orginal / 255
    test_x = test_x_orginal / 255
    train_y = train_y_orginal.reshape(train_y_orginal.shape[0], 1)
    test_y = test_y_orginal.reshape((test_y_orginal.shape[0], 1))

    # 独热编码
    train_y = tf.one_hot(np.squeeze(train_y), 6)
    test_y = tf.one_hot(np.squeeze(test_y), 6)

    return train_x, train_y, test_x, test_y


# 定义占位符
def create_placeholder(n_H, n_W, n_C, n_y):
    X = tf.compat.v1.placeholder(tf.float32, [None, n_H, n_W, n_C])
    Y = tf.compat.v1.placeholder(tf.float32, [None, n_y])

    return X, Y


# 初始化参数
def init_patameters():
    # 只初始化卷积层的W参数，b由tensorflow自动初始化；全连接层也是由tensorflow自动初始化。
    W1 = tf.compat.v1.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.compat.v1.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {
        "W1" : W1,
        "W2" : W2
    }

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    # 获取参数
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # 前向传播
    Z1 =tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool2d(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool2d(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    P = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

    return Z3


# 计算代价函数
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))

    return cost


# 处理mini_batch数据
def random_minibatch_datasets(X, Y, batch_size, seed=0):
    np.random.seed(seed)
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    mini_batches = []

    # 打乱顺序
    X_shuffle = X[permutation, :, :, :]
    Y_shuffle = Y[permutation, :]

    # 切片
    num = m // batch_size
    for i in range(num):
        mini_batch_X = X_shuffle[i * batch_size : (i + 1) * batch_size, :, :, :]
        mini_batch_Y = Y_shuffle[i * batch_size : (i + 1) * batch_size, :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % batch_size !=0:
        mini_batch_X = X_shuffle[num * batch_size:, :, :, :]
        mini_batch_Y = Y_shuffle[num * batch_size:, :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# 启动模型
def model(train_x, train_y, test_x, test_y, learning_rate=0.009, epoch_nums=150, batch_size=64, is_plot=True):
    np.random.seed(1)
    m, n_H, n_W, n_C = train_x.shape
    n_y = train_y.shape[1]
    seed = 3
    num = m // batch_size
    costs = []

    # 定义占位符
    X, Y = create_placeholder(n_H, n_W, n_C, n_y)

    # 初始化参数
    parameters = init_patameters()

    # 前向传播
    Z3 = forward_propagation(X, parameters)

    # 计算损失函数
    cost = compute_cost(Z3, Y)

    # 反向传播和更新参数
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化全局变量
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as session:
        session.run(init)
        for epoch in range(epoch_nums):
            epoch_cost = 0.0
            seed = seed + 1
            mini_batches = random_minibatch_datasets(train_x, train_y.eval(), batch_size, seed)
            for mini_batch in mini_batches:
                mini_batch_X, mini_batch_Y = mini_batch
                _, batch_cost = session.run([optimizer, cost], feed_dict={X : mini_batch_X, Y : mini_batch_Y})
                epoch_cost = epoch_cost + batch_cost / num

            # 打印cost
            if epoch % 5 == 0:
                print("第{}次迭代，cost的值为{}".format(epoch, epoch_cost))

            costs.append(epoch_cost)

        # 绘图
        if is_plot:
            plt.plot(costs)
            plt.xlabel("epoch_num(per five")
            plt.ylabel("cost")
            plt.title("learning_rate=" + str(learning_rate))
            plt.show()

        # 将parameters写进session
        parameters = session.run(parameters)

        # 计算准确率
        prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        accuracy_train = accuracy.eval({X : train_x, Y : train_y.eval()})
        accuracy_test = accuracy.eval({X : test_x, Y : test_y.eval()})

        print("训练集准确率：" + str(accuracy_train))
        print("测试集准确率：" + str(accuracy_test))

    return parameters


# 预测自己的图片
def predict(X, parameters):
    Z3 = forward_propagation(X, parameters)
    prediction = tf.argmax(Z3, 1)
    return prediction


# 加载数据集
train_x, train_y, test_x, test_y = load_datasets()

# 启动模型
parameters = model(train_x, train_y, test_x, test_y)

# 预测自己的图片
for i in range(1, 6):
    # 读取图片
    image = str(i) + ".png"
    image_file = "images/fingers/" + image
    image_read = mpimg.imread(image_file)
    # 显示图片
    plt.imshow(image_read)
    # 预处理图片数据
    image_read = image_read.reshape(1, 64, 64, 3)
    # 预测图片
    prediction = predict(image_read, parameters)
    # 打印图片
    with tf.compat.v1.Session() as session:
        session.run(tf.global_variables_initializer())
        p = session.run(prediction)
        print("第{}张图片是{}".format(i, p))


