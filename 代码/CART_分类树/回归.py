
# refer:https://blog.csdn.net/yujianmin1990/article/details/49864813

# -*- coding:utf-8 -*-
from numpy import *

#读取数据到矩阵

def loadDataSet(filename):
    dataMat = []

    fr = open(filename)

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)  # 将每个数据转成浮点型
        dataMat.append(fltLine)
    return dataMat

#----------------------CART算法----------------------------------#
#“二元切分法”切分数据集：在给定特征和特征值的情况下，通过数组过滤的方式将输入的数据集合切分得到两个子集并返回

def binSplitDataSet(dataSet,feature,value):  #输入：数据集合，待切分特征，该特征的某个值
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]  #注意此处书上源码有误
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

#生成叶节点的函数
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

#误差计算函数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]  #总方差 = 均方差　×　样本数目


#切分函数：根据“总方差”确定最佳二元切分方式（是回归书构建的核心函数！）
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]  #允许的误差下降值
    tolN = ops[1]  #切分的最少样本数量
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:  #当剩余特征值的数目为1的时候，切分结束，直接返回(tolist()是将当前对象转换为python的list对象返回)
        return None, leafType(dataSet)
    m,n = shape(dataSet)   #初始化循环变量
    S = errType(dataSet)
    bestS = inf 
    bestIndex = 0
    bestValue = 0
    #需找最佳二元划分方式： 遍历数据集合中所有特征的所有样本（注意描述顺序），需找到能够使得切分后数据集合效果提升（误差下降最多）对应的特征和特征值返回
    for  featIndex in range(n-1):
        for splitVal  in set(dataSet[:,featIndex].flat): #处书上源码有误(flat()是获得将数组展开为一维的迭代器),这里与dataSet[:,].T.tolist()[0]等效）
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0] < tolN or shape(mat1)[0] < tolN):   #当样本数量小于预设值tolN时，直接进行下一轮判断
                continue
            newS =errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:                         #当误差减小效果不明显（误差下降值未达到于设置值tolS）时，直接返回None和叶节点
        return None, leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): #当样本数量小于预设值tolN时
        return None, leafType(dataSet)
    return bestIndex, bestValue


#构建树：根据函数chooseBestSplit()切分数据，递归的构建二叉树
#输入：数据集合，生成叶节点的函数，误差计算函数，构建树所需要的其他参数元组 
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):   
    #尝试将数据集合分成两部分
    #切分函数按chooseBestSplit()函数进行
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:   # 如果满足停止条件，chooseBestSplit()函数返回None和某类模型的值（回归树：常数；模型树：线性方程）
        return val
    #将数据集合分成左子树、右子树两部分
    retTree= {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    #继续递归左子树和右子树
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


#树剪枝：
#伪代码
#基于已有的树切分测试数据：
#     如果存在任一子集是一棵树，则在该子集地柜剪枝过程
#     计算将当前两个叶节点合并后的误差
#     计算不合并的误差
#     如果合并的误差会降低的话，就将叶节点合并

# 测试输入变量是否位一棵树(也就相当于测试当前处理的节点是否为叶节点)
def isTree(obj):
    return (type(obj).__name__ == 'dict')

# 递归函数，从上往下遍历树直到叶节点为止，然后对树做塌陷处理（即返回树的平均值）
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

#剪枝主函数：　
#输入：　待剪枝的树，剪枝所需的测试数据
def prune(tree,testData):
    #测试集合为空，则直接对树进行塌陷处理
    if shape(testData)[0] == 0:  
        return getMean(tree)
    #测试集合非空，则递归地调用prune()函数对测试数据进行切分
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet ,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet ,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2)) #power()用于求幂
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print "merging"
            return  treeMean
        else:
            return tree
    else:
        return tree

