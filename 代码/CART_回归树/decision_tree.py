# refer: https://github.com/RRdmlearning/Decision-Tree

import csv
from collections import defaultdict
import pydotplus
import numpy as np


# Important part

# 树的结构由它存储
class Tree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, summary=None, data=None):

        # 最优分割列
        self.col = col
        # 当前列内选择的分割点
        self.value = value

        # 左右(是否为此值) 分支
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

        # 此节点上的分类结果（叶子节点才会设置）
        self.results = results

        # 存储叶子节点上的数据信息
        self.data = data

        # 类似于日志信息,画图时需要
        self.summary = summary



# 统计y的类别个数
def calculleftmean(datas):
    list = [i[-1] for i in datas]
    return np.mean(list)


# 计算回归，意味着连续数值类型
def MSE(rows):
    list = [i[-1] for i in rows]
    list_mean = np.mean(list)
    list_var = [(list[i] - list_mean)**2 for i in range(len(rows))]
    mse = sum(list_var)
    return mse

def splitDatas(rows, value, column):
    # 根据条件分离数据集(splitDatas by value,column)
    # return 2 part(list1,list2)

    list1 = []
    list2 = []

    # 如果为数值类型
    if (isinstance(value, int) or isinstance(value, float)):  # for int and float type
        for row in rows:
            # 二分数据集
            if (row[column] >= value):
                list1.append(row)
            else:
                list2.append(row)
    else:  # for String type
        for row in rows:
            if row[column] == value:
                list1.append(row)
            else:
                list2.append(row)

    return (list1, list2)


def chose_best_feature(datas,evaluationFunction=MSE):

    currentGain = evaluationFunction(datas)

    column_length = len(datas[0])
    row_length = len(datas)

    # 初始化
    best_gain = 0.0
    best_value = None
    best_set = None

    # choose the best gain
    for col in range(column_length - 1):

        col_value_set = set([x[col] for x in datas])

        for value in col_value_set:
            list1, list2 = splitDatas(datas, value, col)
            p = len(list1) / row_length
            gain = currentGain - p * evaluationFunction(list1) - (1 - p) * evaluationFunction(list2)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, value)
                best_set = (list1, list2)

    return currentGain,best_gain,best_value,best_set


# 递归建立决策树,当gain = 0 时停止递归
def buildDecisionTree(rows,evaluationFunction=MSE,print_summary = False):

    rows_length = len(rows)

    # 选择最由列的分割点
    currentGain,best_gain, best_value, best_set = chose_best_feature(rows)

    dcY = {'impurity': '%.3f' % currentGain, 'samples': '%d' % rows_length}

    # 打印构建过程中的日志信息，默认否
    if print_summary:
        # 类似与构建日志，会存储于树节点的的信息中
        print(dcY)

    # 递归
    if best_gain > 0:

        trueBranch = buildDecisionTree(best_set[0],print_summary = False)
        falseBranch = buildDecisionTree(best_set[1],print_summary = False)

        return Tree(col=best_value[0], value=best_value[1], trueBranch=trueBranch, falseBranch=falseBranch,summary=dcY)
    else:
        return Tree(results=calculleftmean(rows),summary=dcY,data=rows)


# 剪枝, when gain < mini Gain，合并(merge the trueBranch and the falseBranch)
def prune(tree, miniGain,evaluationFunction=MSE):

    # 递归寻找其叶子节点
    if tree.trueBranch.results == None: prune(tree.trueBranch, miniGain)
    if tree.falseBranch.results == None: prune(tree.falseBranch, miniGain)

    # 叶子节点的父亲节点，因为result只在叶子节点上有值。
    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = len(tree.trueBranch.data)
        len2 = len(tree.falseBranch.data)
        len3 = len(tree.trueBranch.data + tree.falseBranch.data)
        p = float(len1) / (len1 + len2)
        gain = evaluationFunction(tree.trueBranch.data + tree.falseBranch.data) - p * evaluationFunction(
            tree.trueBranch.data) - (1 - p) * evaluationFunction(tree.falseBranch.data)

        if (gain < miniGain):
            tree.data = tree.trueBranch.data + tree.falseBranch.data
            tree.results = calculleftmean(tree.data)

            tree.trueBranch = None
            tree.falseBranch = None

# 也只是针对单行数据
def classify(data, tree):
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(data, branch)


#下面是辅助代码画出树
#Unimportant part
#plot tree and load data
def plot(decisionTree):
    """Plots the obtained decision tree. """

    def toString(decisionTree, indent=''):

        if decisionTree.results != None:  # leaf node
            return str(decisionTree.results)
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %s?' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s?' % (szCol, decisionTree.value)
            trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def loadCSV(file):
    """Loads a CSV file and converts all floats and ints into basic datatypes."""
    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt'))
    dcHeader = {}
    if bHeader:
        lsHeader = next(reader)
        for i, szY in enumerate(lsHeader):
                szCol = 'Column %d' % i
                dcHeader[szCol] = str(szY)
    return dcHeader, [[convertTypes(item) for item in row] for row in reader]



bHeader = True
# the bigger example
dcHeadings, trainingData = loadCSV('fishiris.csv') # demo data from matlab
decisionTree = buildDecisionTree(trainingData)
result = plot(decisionTree)

# 减枝
prune(decisionTree, 0.4) # notify, when a branch is pruned (one time in this example)
result = plot(decisionTree)

