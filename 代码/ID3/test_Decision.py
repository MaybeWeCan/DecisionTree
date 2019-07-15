from DecisionTree import *
import decisionTreePlot as dtPlot


# 测试准确性
def accuracy(a, b):
    c = [a[i] - b[i] for i in range(len(a))]
    correct_number = len([i for i in c if i == 0])
    the_accuracy = float(correct_number / len(a))
    return the_accuracy

if __name__ == '__main__':

    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lenses.txt')

    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses = [i[1:] for i in lenses]

    # 测试数据，比较懒...，没有另外找
    test_data = [i[:-1] for i in lenses]

    # 得到数据的对应的 Labels
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesLabels = ['prescript', 'astigmatic', 'tearRate']
    test_labels = copy.deepcopy(lensesLabels)  # 深拷贝就是:labels_copy和lables撇清关系

    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    my_tree = DecisionTree()
    lensesTree = my_tree.train(lenses,lensesLabels,test_labels,test_data,1,True)
    print(lensesTree)

    # 画图可视化展现
    #dtPlot.createPlot(lensesTree)

    # 测试代码
    classLabel = my_tree.predict(lensesTree,test_labels,test_data)
    print(classLabel)