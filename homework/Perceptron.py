from functools import reduce

class Perceptron(object):
    # 初始化感知器，设置输入参数的个数，以及激活函数。激活函数的类型为double -> double
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]  # 权重向量初始化为0
        self.bias = 0.0  # 偏置项初始化为0

    # 打印学习到的权重、偏置项
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    # 输入向量，输出感知器的计算结果
    def predict(self, input_vec):
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        result_map = map(lambda x,w: x * w,input_vec, self.weights)
        result_reduce = reduce(lambda a, b: a + b,result_map, 0.0)
        print(list(result_map))


        return self.activator(result_reduce + self.bias)

    # 输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate) #一次迭代

    # 一次迭代，把所有的训练数据过一遍
    def _one_iteration(self, input_vecs, labels, rate):
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:  # 对每个样本，按照感知器规则更新权重
            output = self.predict(input_vec)  # 计算感知器在当前权重下的输出
            self._update_weights(input_vec, output, label, rate)  # 更新权重

    # 按照感知器规则更新权重
    def _update_weights(self, input_vec, output, label, rate):
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights =map(
            lambda x, w: w + rate * delta * x,
            input_vec, self.weights)
        self.bias += rate * delta  # 更新bias

# 定义激活函数f
def f(x):
    return 1 if x > 0 else 0

# 基于and真值表构建训练数据
def get_training_dataset():# 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels

# 使用and真值表训练感知器
def train_and_perceptron():
    p = Perceptron(2, f)  # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    input_vecs, labels = get_training_dataset() # 构建训练数据
    #假设迭代10次，收敛w和b的最优值
    p.train(input_vecs, labels, 10, 0.1) # 训练，迭代10轮, 学习速率为0.1
    return p  # 返回训练好的感知器

if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    #print(list(and_perception.weights))
    #print(and_perception.bias)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
