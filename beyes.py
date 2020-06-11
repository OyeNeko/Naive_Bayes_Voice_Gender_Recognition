# -*- coding:utf-8 -*-

import numpy as np
import csv
import matplotlib.pyplot as plt

""""对男女声音进行辨别"""

def load_data_set(file_name):
    """
    :param file_name: 文件名字
    :return

    train_mat：离散化的训练数据集
    train_classes： 训练数据集所属的分类
    test_mat：离散化的测试数据集
    test_classes：测试数据集所述的分类
    label_name：特征的名称
    """
    data_mat = []
    with open(file_name) as file_obj:
        voice_reader = csv.DictReader(file_obj)
        list_class = []
        # 文件头
        label_name = list(voice_reader.fieldnames)
        num = len(label_name) - 1

        for line in voice_reader.reader:
            data_mat.append(line[:num])
            gender = 1 if line[-1] == 'male' else 0
            list_class.append(gender) #性别数组

        # 求每一个特征的平均值
        data_mat = np.array(data_mat).astype(float)
        count_vector = np.count_nonzero(data_mat, axis=0)#统计不为零的个数
        sum_vector = np.sum(data_mat, axis=0)
        mean_vector = sum_vector / count_vector

        # 数据缺失的地方 用 平均值填充
        for row in range(len(data_mat)):
            for col in range(num):
                if data_mat[row][col] == 0.0:
                    data_mat[row][col] = mean_vector[col]

        # 将数据连续型的特征值离散化处理
        min_vector = data_mat.min(axis=0)
        max_vector = data_mat.max(axis=0)
        diff_vector = max_vector - min_vector
        diff_vector /= N
        new_data_set = []
         
        for i in range(len(data_mat)):
            line = np.array((data_mat[i] - min_vector) / diff_vector).astype(int)
            new_data_set.append(line)
        data_mat_discretized=np.array(new_data_set)
        """ #可视化数据集
        plt.figure()
        pltvalues = np.array(data_mat[:, 0]).flatten()
        plt.plot(pltvalues,linewidth=1)#参数linewidth决定了plot()绘制的线条的粗细。
        plt.title('Initial_meanfreq(N=%d)'%N,fontsize=24) #函数title()给图表指定标题
        plt.xlabel('num',fontsize=14) #函数xlabel()和ylabel()让你能够为每条轴设置标题
        plt.ylabel('value',fontsize=14)
        plt.tick_params(axis='both',labelsize=14) #而函数tick_params()设置刻度的样式，其中指定的实参将影响x轴和y轴上的刻度（axes='both'），并将刻度标记的字号设置为14（labelsize=14）。
        #可视化数据集
        plt.figure()
        pltvalues = np.array(data_mat_discretized[:, 0]).flatten()       
        plt.plot(pltvalues,linewidth=1)#参数linewidth决定了plot()绘制的线条的粗细。
        plt.title('discretized_meanfreq(N=%d)'%N,fontsize=24) #函数title()给图表指定标题
        plt.xlabel('num',fontsize=14) #函数xlabel()和ylabel()让你能够为每条轴设置标题
        plt.ylabel('value',fontsize=14)
        plt.tick_params(axis='both',labelsize=14) #而函数tick_params()设置刻度的样式，其中指定的实参将影响x轴和y轴上的刻度（axes='both'），并将刻度标记的字号设置为14（labelsize=14）。
        """
        
        # 随机划分数据集为训练集 和 测试集
        test_set = list(range(len(new_data_set)))
        train_set = []
        for i in range(2000):
            random_index = int(np.random.uniform(0, len(test_set)))
            train_set.append(test_set[random_index])
            del test_set[random_index]

        # 训练数据集
        train_mat = []
        train_classes = []
        for index in train_set:
            train_mat.append(new_data_set[index])
            train_classes.append(list_class[index])

        # 测试数据集
        test_mat = []
        test_classes = []
        for index in test_set:
            test_mat.append(new_data_set[index])
            test_classes.append(list_class[index])

    return train_mat, train_classes, test_mat, test_classes, label_name


def native_bayes(train_matrix, list_classes):
    """
    :param train_matrix: 训练样本矩阵
    :param list_classes: 训练样本分类向量
    :return:p_1_class 任一样本分类为1的概率  p_feature,p_1_feature 分别为给定类别的情况下所以特征所有取值的概率
    """
    # 训练样本个数
    num_train_data = len(train_matrix)
    num_feature = len(train_matrix[0])
    # 分类为1的样本占比
    p_1_class = sum(list_classes) / float(num_train_data)
    p_0_class= 1 - p_1_class
    n = N+1  #离散程度
    list_classes_1 = []
    train_data_1 = []
    list_classes_0 = []
    train_data_0 = []
    for i in list(range(num_train_data)):
        if list_classes[i] == 1:#为男性取出
            list_classes_1.append(i)
            train_data_1.append(train_matrix[i])
        else:
            list_classes_0.append(i)#为女性
            train_data_0.append(train_matrix[i])

    # 分类为1 情况下的各特征的概率
    train_data_1 = np.matrix(train_data_1)
    p_1_feature = {}
    for i in list(range(num_feature)):
        feature_values = np.array(train_data_1[:, i]).flatten()
        # 避免某些特征值概率为0 影响总体概率，每个特征值最少个数为1
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))#先得到平均值，然后取对数
        p_1_feature[i] = p

    # 分类为0 情况下的各特征的概率
    train_data_0 = np.matrix(train_data_0)
    p_0_feature = {}
    for i in list(range(num_feature)):
        feature_values = np.array(train_data_0[:, i]).flatten()
        # 避免某些特征值概率为0 影响总体概率，每个特征值最少个数为1
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))#先得到平均值，然后取对数
        p_0_feature[i] = p

    return p_1_feature, p_1_class, p_0_feature, p_0_class


def classify_bayes(test_vector, p_1_feature, p_1_class, p_0_feature, p_0_class):
    """
    :param test_vector: 要分类的测试向量
    :param p_1_feature: 类别为1的情况下所有特征所有取值的概率
    :param p_1_class: 任一样本分类为1的概率
    :param p_0_feature: 类别为0的情况下所有特征所有取值的概率
    :param p_0_class: 任一样本分类为0的概率
    :return: 1 表示男性 0 表示女性
    """
    # 计算每个分类的概率(概率相乘取对数 = 概率各自对数相加)
    sum_1 = 0.0
    sum_0 = 0.0
    for i in list(range(len(test_vector))):
        sum_1 += p_1_feature[i][test_vector[i]]
        sum_0 += p_0_feature[i][test_vector[i]]

    p1 = sum_1 + np.log(p_1_class)
    p0 = sum_0 + np.log(p_0_class)
    if p1 > p0:
        return 1
    else:
        return 0


def test_bayes():
    file_name = 'data/voice.csv'
    train_mat, train_classes, test_mat, test_classes, label_name = load_data_set(file_name)

    p_1_feature, p_1_class, p_0_feature, p_0_class = native_bayes(train_mat, train_classes)

    count = 0.0
    correct_count = 0.0
    male_accurate=0.0
    male_wrong=0.0
    female_accurate=0.0
    female_wrong=0.0
    male_num=0.0
    female_num=0.0
    for i in list(range(len(test_mat))):
        test_vector = test_mat[i]
        result = classify_bayes(test_vector,p_1_feature, p_1_class, p_0_feature, p_0_class)
        if test_classes[i] == 1:#是男性
            male_num += 1
            if result == test_classes[i]:
                male_accurate +=1
            else:
                male_wrong +=1
        else:#是女性
            female_num += 1
            if result == test_classes[i]:
                female_accurate += 1
            else:
                female_wrong += 1
    male_accurate_rate.append(male_accurate/male_num)
    male_wrong_rate.append(male_wrong/male_num)
    female_accurate_rate.append(female_accurate/female_num)
    female_wrong_rate.append(female_wrong/female_num)
    total_accurate_rate.append((male_accurate+female_accurate)/(male_num+female_num))
    

j=0
N=10
male_accurate_rate=[]
male_wrong_rate=[]
female_accurate_rate=[]
female_wrong_rate=[]
total_accurate_rate=[]

for j in range(30):
    N = N+2
    test_bayes()
for j in range(30):

    print("第%d次：" % (j+1))
    print("男性正确率：", male_accurate_rate[j])
    print("男性错误率：", male_wrong_rate[j])
    print("女性正确率：", female_accurate_rate[j])
    print("女性错误率：", female_wrong_rate[j])
    print("总正确率：", total_accurate_rate[j])
#可视化数据集
plt.figure()
plt.plot(total_accurate_rate,linewidth=1)#参数linewidth决定了plot()绘制的线条的粗细。
plt.title('total_accurate_rate',fontsize=24) #函数title()给图表指定标题
plt.xlabel('times',fontsize=14) #函数xlabel()和ylabel()让你能够为每条轴设置标题
plt.ylabel('rate',fontsize=14)
plt.tick_params(axis='both',labelsize=14) #而函数tick_params()设置刻度的样式，其中指定的实参将影响x轴和y轴上的刻度（axes='both'），并将刻度标记的字号设置为14（labelsize=14）。

plt.figure()

plt.plot(male_accurate_rate,linewidth=1)#参数linewidth决定了plot()绘制的线条的粗细。
plt.title('male_accurate_rate',fontsize=24) #函数title()给图表指定标题
plt.xlabel('times',fontsize=14) #函数xlabel()和ylabel()让你能够为每条轴设置标题
plt.ylabel('rate',fontsize=14)
plt.tick_params(axis='both',labelsize=14) #而函数tick_params()设置刻度的样式，其中指定的实参将影响x轴和y轴上的刻度（axes='both'），并将刻度标记的字号设置为14（labelsize=14）。


plt.figure()

plt.plot(female_accurate_rate,linewidth=1)#参数linewidth决定了plot()绘制的线条的粗细。
plt.title('female_accurate_rate',fontsize=24) #函数title()给图表指定标题
plt.xlabel('times',fontsize=14) #函数xlabel()和ylabel()让你能够为每条轴设置标题
plt.ylabel('rate',fontsize=14)
plt.tick_params(axis='both',labelsize=14) #而函数tick_params()设置刻度的样式，其中指定的实参将影响x轴和y轴上的刻度（axes='both'），并将刻度标记的字号设置为14（labelsize=14）。
plt.show()
