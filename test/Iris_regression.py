from sklearn import datasets #数据来源
from sklearn.metrics import classification_report #分类报告
from sklearn.model_selection import train_test_split #数据分割
from sklearn import svm #模型选择
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test_num = 45

def IrisTrain_svm():     #预测第四个特征
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    s = svm.SVR()
    s.fit(x_train,y_train)
    test_x = np.c_[x_test[:,0:3],y_test]     #合并数组 也可以用np.stack()
    iris_predict_f4 = s.predict(test_x)  # 预测
    iris_real_f4 = x_test[:,3]
    return iris_predict_f4,iris_real_f4,x_test,y_test,s

def IrisTrain_corr(a,b):
    a_s = pd.Series(a)
    b_s = pd.Series(b)
    cor = a_s.corr(b_s)
    return cor

if __name__ == '__main__':
    lables = ["德国鸢尾", "荷兰鸢尾", "网脉鸢尾"]

    iris_predict_s, iris_real, iris_Xtest_s, iris_Ytest_s, s= IrisTrain_svm()
    x_axis = np.arange(0,45,1)
    plt.plot(x_axis,iris_real,'.',color='red')
    plt.plot(x_axis,iris_predict_s,'.',color='blue')
    plt.show()
    cor1 = IrisTrain_corr(iris_predict_s,iris_real)
    print("预测值与真实值得相关度：{:.2%}".format(cor1))