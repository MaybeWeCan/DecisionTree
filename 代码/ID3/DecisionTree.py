from math import log
import operator # operator模块是用c实现的，所以执行速度比python代码快
from collections import Counter
import numpy as np
import copy

class DecisionTree:
    #def __init__(self):

    def calcShannonEnt(self,dataSet):
        """
        Desc：
            calculate Shannon entropy -- 计算给定数据集的香农熵
        Args:
            dataSet -- 数据集
        Returns:
            shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望
        """
        numEntries = len(dataSet)

        labelCounts = {}
        # the the number of unique elements and their occurance
        for featVec in dataSet:
            # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
            currentLabel = featVec[-1]
            # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1

        shannonEnt = 0.0

        for key in labelCounts:
            # 使用所有类标签的发生频率计算类别出现的概率。
            prob = float(labelCounts[key]) / numEntries
            # log base 2
            # 计算香农熵，以 2 为底求对数
            shannonEnt -= prob * log(prob, 2)

        return shannonEnt

    # 离散值：该函数才是那个最秒的...解决了递归三个要求之一：规模越来越小，其行、列同时选择的方法很巧妙
    def splitDataSet(self,dataSet, index, value):
        """
        Desc：
            划分数据集
            splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
            就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
        Args:
            dataSet  -- 数据集                 待划分的数据集
            index -- 表示每一行的index列        划分数据集的特征
            value -- 表示index列对应的value值   需要返回的特征的值。
        Returns:
            index 列为 value 的数据集【该数据集需要排除index列】
        """
        retDataSet = [data[:index] + data[index + 1:] for data in dataSet for i, v in enumerate(data) if
                      i == index and v == value]

        return retDataSet

    # 连续值的分割
    def splitDataSetForSeries(self,dataSet, axis, value):

        """
        按照给定的数值，将数据集分为不大于和大于两部分
        :param dataSet: 要划分的数据集
        :param i: 特征值所在的下标
        :param value: 划分值
        :return:
        """
        # 用来保存不大于划分值的集合
        eltDataSet = []
        # 用来保存大于划分值的集合
        gtDataSet = []
        # 进行划分，保留该特征值
        for feat in dataSet:
            if feat[axis] <= value:
                eltDataSet.append(feat)
            else:
                gtDataSet.append(feat)

        return eltDataSet, gtDataSet

    # 求单列连续特征信息增益的函数
    def calcInfoGainForSeries(self,dataSet, i, baseEntropy):
        """
        计算连续值的信息增益
        :param dataSet:整个数据集
        :param i: 对应的特征值下标
        :param baseEntropy: 基础信息熵
        :return: 返回一个信息增益值，和当前的划分点
        """

        # 记录最大的信息增益
        maxInfoGain = 0.0

        # 最好的划分点
        bestMid = -1

        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]

        # 得到分类列表
        classList = [example[-1] for example in dataSet]

        dictList = dict(zip(featList, classList))

        # 将其从小到大排序，按照连续值的大小排列
        sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))

        # 计算连续值有多少个
        numberForFeatList = len(sortedFeatList)

        # 二分隔，所以用相邻两个数的中间值分割
        midFeatList = [round((sortedFeatList[k][0] + sortedFeatList[k + 1][0]) / 2.0, 3) for k in
                       range(numberForFeatList - 1)]

        # 计算出各个划分点信息增益
        for mid in midFeatList:

            # 将连续值划分为不大于当前划分点和大于当前划分点两部分
            eltDataSet, gtDataSet = self.splitDataSetForSeries(dataSet, i, mid)

            # 计算两部分的特征值熵和权重的乘积之和
            newEntropy = float(len(eltDataSet)) / float(len(sortedFeatList)) * float(
                self.calcShannonEnt(eltDataSet)) + float(len(gtDataSet)) / float(len(sortedFeatList)) * float(
                self.calcShannonEnt(gtDataSet))

            # 计算出信息增益
            infoGain = baseEntropy - newEntropy
            # print('当前划分值为：' + str(mid) + '，此时的信息增益为：' + str(infoGain))
            if infoGain > maxInfoGain:
                bestMid = mid
                maxInfoGain = infoGain

        return maxInfoGain, bestMid

    # 求单列离散特征信息增益的函数
    def calcInfoGain(self,dataSet, featList, i, baseEntropy):
        """
        计算信息增益
        :param dataSet: 数据集
        :param featList: 当前特征列表
        :param i: 当前特征值下标
        :param baseEntropy: 基础信息熵
        :return:
        """
        # 将当前特征唯一化，也就是说当前特征值中共有多少种
        uniqueVals = set(featList)

        # 新的熵，代表当前特征值的熵
        newEntropy = 0.0

        # 遍历现在有的特征的可能性
        for value in uniqueVals:
            # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
            subDataSet = self.splitDataSet(dataSet, i, value)
            # 计算出权重
            prob = float(len(subDataSet)) / float(len(dataSet))
            # 计算出当前特征值的熵
            newEntropy += prob * self.calcShannonEnt(subDataSet)

        # 计算出“信息增益”
        infoGain = baseEntropy - newEntropy

        return infoGain

    def chooseBestFeatureToSplit(self,dataSet):
        """
        Desc:
            选择切分数据集的最佳特征
        Args:
            dataSet -- 需要切分的数据集
        Returns:
            bestFeature -- 切分数据集的最优的特征列
        """
        # 得到数据的特征值总数
        numFeatures = len(dataSet[0]) - 1
        print(numFeatures)

        # 计算初始香农熵，单纯基于标签
        baseEntropy = self.calcShannonEnt(dataSet)

        # 基础信息增益为0.0
        bestInfoGain = 0

        # 最好特征值的index
        bestFeature = -1

        # 标记当前最好的特征值是不是连续值
        flagSeries = 0

        # 如果是连续值的话，用来记录连续值的划分点
        bestSeriesMid = 0.0

        # 遍历每一个特征
        for i in range(numFeatures):

            # 对当前第i列的特征进行统计
            # feature_count = Counter([data[i] for data in dataSet])
            featList = [example[i] for example in dataSet]

            # 判断是否连续离散
            series_flag = len(set(featList))

            # 可以自己调,当然这些直接写死很麻瓜。
            # 参考的代码其函数的写法个人不赞同，实际情况我们也会去用独热编码的，单纯字符串判断不了
            if series_flag < 10:
                # 离散求解信息增益
                infoGain = self.calcInfoGain(dataSet, featList, i, baseEntropy)
                print(infoGain)
            else:
                # 连续数值求解信息增益
                infoGain,bestMid = self.calcInfoGainForSeries(dataSet, i, baseEntropy)
                flagSeries = 1

            # 如果当前的信息增益比原来的大
            if infoGain > bestInfoGain:
                # 最好的信息增益
                bestInfoGain = infoGain

                # 新的最好的用来划分的特征值
                bestFeature = i


            if flagSeries:
                bestSeriesMid = bestMid

            # print('信息增益最大的特征为：' + labels[bestFeature])
            if flagSeries:
                return bestFeature, bestSeriesMid
            else:
                return bestFeature

    # 这个函数用于预剪枝和后剪枝
    def testingMajor(self,major,data_test):
        error = 0.0
        for i in range(len(data_test)):
            if major != data_test[i][-1]:
                error += 1
        return float(error)

    def majorityCnt(self,classList):
        """
        Desc:
            选择出现次数最多的一个结果
        Args:
            classList label列的集合
        Returns:
            bestFeature 最优的特征列
        """
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
        '''sorted用法比较'''
        # sorted(students, key=lambda student : student[2])
        # sorted(students, key=operator.itemgetter(2))
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


    # 利用训练好的树分类，同样采用递归,这里我真的是无语了，思维误区害死人
    def classify(self,inputTree, featLabels, testVec):
        """
        Desc:
            对新数据进行分类
        Args:
            inputTree  -- 已经训练好的决策树模型
            featLabels -- Feature标签对应的名称，不是目标变量
            testVec    -- 测试输入的数据，注意注意！！！！！ 一维的列表......
        Returns:
            classLabel -- 分类的结果值，需要映射label才能知道名称
        """
        # 获取tree的根节点对于的key值
        firstStr = list(inputTree.keys())[0]

        # 通过key得到根节点对应的value
        secondDict = inputTree[firstStr]

        # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
        featIndex = featLabels.index(firstStr)

        # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
        key = testVec[featIndex]

        valueOfFeat = secondDict[key]

        # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
        if isinstance(valueOfFeat, dict):
            classLabel = self.classify(valueOfFeat, featLabels, testVec)
        else:
            classLabel = valueOfFeat
        return classLabel

    # 这个函数专门用于"后剪枝",错误率计算
    def testing(self,myTree, data_test, lists):
        # 这里输入的labels不是全部的特征名称
        # 这里输入的data_test不带有全部的特征名称

        error = 0.0
        for i in range(len(data_test)):
            if self.classify(myTree, lists, data_test[i]) != data_test[i][-1]:  # 如果预测结果与验证数据的类别标签不一致
                error += 1  # 那么错误数就+1
        return float(error)

    '''
    程序思路：
        1. 以递归为主体，循环当然也可以，但是没有递归简洁，可如果是针对大规模数据呢？？
        2. 程序出口( 出口位置在程序前后中会有影响 )：  （1）特征为空，返回此时的类别
                     （2） 所有实例均属于一类，返回此时的类别
           解释：不能递归的原因有两个，我这个节点达到目的，不需要继续递归了，我这个节点即使可以递归，但是没有多的特征允许了。
        3. 树结构存储在字典内。个人以前还一直以为是指针来着....,但是在字典里面，越想越觉得秒啊    
    '''

    def my_createTree(self,dataSets,feature_name,test_feature_name,test_data,theta,is_post):

        """
        Desc:
            创建决策树
        Args:
            dataSet -- 要创建决策树的训练数据集
            feature_name -- 训练数据集中特征对应的含义的labels，不是目标变量
        Returns:
            myTree -- 创建完成的决策树
        """
        # 第一个出口，节点内全部为一个样本
        labels = [i[-1] for i in dataSets]

        if labels.count(labels[0]) == len(labels):
            return labels[0]

        # 第二个出口，没有特征可供继续分支
        if len(dataSets[0]) == 1:
            return self.majorityCnt(labels)

        # 得到最好特征的名称
        bestFeatLabel = ''

        # 正常开始： 1. 选取特征,单纯基于  2. 特征分支递归
        # 1 best_feature: index

        # 就这一行，开始没错，后来错了？？？ bestFeat = -1
        bestFeat = self.chooseBestFeatureToSplit(dataSets)

        # 得到分叉点信息
        bestFeatLabel = feature_name[bestFeat]
        # 离散值标志
        flagSeries = 0

        # 2分支开始
        # 构造存储树的字典
        myTree = {bestFeatLabel: {}}

        # 得到当前特征标签的所有可能值
        featValues = [example[bestFeat] for example in dataSets]


        # 离散值处理
        print("离散")
        # 将本次划分的特征值从列表中删除掉
        del (feature_name[bestFeat])

        # 唯一化，去掉重复的特征值
        uniqueVals = set(featValues)

        # 遍历所有的特征值
        for value in uniqueVals:
            # 得到剩下的特征标签
            subLabels = feature_name[:]
            # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
            subTree = self.my_createTree(self.splitDataSet(dataSets,bestFeat,value), subLabels,test_feature_name,test_data,theta,is_post)
            # 将子树归到分叉处下
            myTree[bestFeatLabel][value] = subTree

        if is_post:
            # 继续分错误的损失 - 部分错误的损失 > 阈值
            if self.testing(myTree, test_data, test_feature_name) - self.testingMajor(self.majorityCnt(labels),
                                                                                      test_data) > theta:
                return self.majorityCnt(labels)
                # 实现后剪枝操作
                # 无视当前的myThree,直接返回一个叶子节点,等效于实现了REP后剪枝

        #后减枝操作
        return myTree

    # 为了同步，这里简单包装
    def train(self,dataSets,feature_name,test_labels,test_data,theta=1,is_post=True):
        result = self.my_createTree(dataSets,feature_name,test_labels,test_data,theta,is_post)
        return result

    # 适应一、二维列表输入
    def predict(self,inputTree, featLabels, testVec):
        """
        Desc:
            对新数据进行分类
        Args:
            inputTree  -- 已经训练好的决策树模型
            featLabels -- Feature标签对应的名称，不是目标变量
            testVec    -- 测试输入的数据，注意注意！！！！！ 一维的列表......
        Returns:
            classLabel -- 分类的结果值，需要映射label才能知道名称
        """

        labels = []

        #
        if isinstance(testVec[0],list):
            for data in testVec:
                label = self.classify(inputTree, featLabels, data)
                labels.append(label)

        else:
            label = self.classify(inputTree, featLabels,testVec)
            labels.append(label)

        return labels
