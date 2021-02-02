from sklearn import datasets #数据来源
from sklearn.metrics import classification_report #分类报告
from sklearn.model_selection import train_test_split #数据分割
from sklearn import svm #模型选择
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

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

def IrisTrain_knn():     #预测第四个特征
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    knn = KNeighborsRegressor()
    knn.fit(x_train,y_train)
    test_x = np.c_[x_test[:,0:3],y_test]     #合并数组 也可以用np.stack()
    iris_predict_f4 = knn.predict(test_x)  # 预测
    iris_real_f4 = x_test[:,3]
    return iris_predict_f4,iris_real_f4,x_test,y_test,knn

def IrisTrain_mlp():     #预测第四个特征
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    mlp = MLPRegressor()
    mlp.fit(x_train,y_train)
    test_x = np.c_[x_test[:,0:3],y_test]     #合并数组 也可以用np.stack()
    iris_predict_f4 = mlp.predict(test_x)  # 预测
    iris_real_f4 = x_test[:,3]
    return iris_predict_f4,iris_real_f4,x_test,y_test,mlp

def IrisTrain_sgd():     #预测第四个特征
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    sgd = SGDRegressor()
    sgd.fit(x_train,y_train)
    test_x = np.c_[x_test[:,0:3],y_test]     #合并数组 也可以用np.stack()
    iris_predict_f4 = sgd.predict(test_x)  # 预测
    iris_real_f4 = x_test[:,3]
    return iris_predict_f4,iris_real_f4,x_test,y_test,sgd

def IrisTrain_Decisiontree():     #预测第四个特征
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    dt = tree.DecisionTreeRegressor()
    dt.fit(x_train,y_train)
    test_x = np.c_[x_test[:,0:3],y_test]     #合并数组 也可以用np.stack()
    iris_predict_f4 = dt.predict(test_x)  # 预测
    iris_real_f4 = x_test[:,3]
    return iris_predict_f4,iris_real_f4,x_test,y_test,sgd

def IrisTrain_RandomForest():     #预测第四个特征
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    rf = RandomForestRegressor()
    rf.fit(x_train,y_train)
    test_x = np.c_[x_test[:,0:3],y_test]     #合并数组 也可以用np.stack()
    iris_predict_f4 = rf.predict(test_x)  # 预测
    iris_real_f4 = x_test[:,3]
    return iris_predict_f4,iris_real_f4,x_test,y_test,rf

def IrisTrain_corr(a,b):
    a_s = pd.Series(a)
    b_s = pd.Series(b)
    cor = a_s.corr(b_s)
    return cor

if __name__ == '__main__':
    lables = ["德国鸢尾", "荷兰鸢尾", "网脉鸢尾"]
    x_axis = np.arange(0,45,1)

    iris_predict_s, iris_real_s, iris_Xtest_s, iris_Ytest_s, s= IrisTrain_svm()
    cor_svm = IrisTrain_corr(iris_predict_s,iris_real_s)
    print("预测值与真实值得相关度：{:.2%}".format(cor_svm))
    # plt.plot(x_axis,iris_real_s,'.',color='red')
    # plt.plot(x_axis,iris_predict_s,'.',color='blue')
    plt.figure()
    plt.scatter(x_axis,iris_real_s,color='red')
    plt.scatter(x_axis,iris_predict_s,color='blue')
    plt.title("SVM")
    plt.xlabel('the num of object')
    plt.ylabel('the value of feature')
    plt.show()

    iris_predict_knn, iris_real_knn, iris_Xtest_knn, iris_Ytest_knn, knn= IrisTrain_knn()
    cor_knn = IrisTrain_corr(iris_predict_knn,iris_real_knn)
    print("预测值与真实值得相关度：{:.2%}".format(cor_knn))
    plt.figure()
    plt.scatter(x_axis,iris_real_knn,color='red')
    plt.scatter(x_axis,iris_predict_knn,color='blue')
    plt.title("KNN")
    plt.xlabel('the num of object')
    plt.ylabel('the value of feature')
    plt.show()

    iris_predict_mlp, iris_real_mlp, iris_Xtest_mlp, iris_Ytest_mlp, mlp= IrisTrain_mlp()
    cor_mlp = IrisTrain_corr(iris_predict_mlp,iris_real_mlp)
    print("预测值与真实值得相关度：{:.2%}".format(cor_mlp))
    plt.figure()
    plt.scatter(x_axis,iris_real_mlp,color='red')
    plt.scatter(x_axis,iris_predict_mlp,color='blue')
    plt.title("MLP")
    plt.xlabel('the num of object')
    plt.ylabel('the value of feature')
    plt.show()

    iris_predict_sgd, iris_real_sgd, iris_Xtest_sgd, iris_Ytest_sgd, sgd= IrisTrain_sgd()
    cor_sgd = IrisTrain_corr(iris_predict_sgd,iris_real_sgd)
    print("预测值与真实值得相关度：{:.2%}".format(cor_sgd))
    plt.figure()
    plt.scatter(x_axis,iris_real_sgd,color='red')
    plt.scatter(x_axis,iris_predict_sgd,color='blue')
    plt.title("SGD")
    plt.xlabel('the num of object')
    plt.ylabel('the value of feature')
    plt.show()

    iris_predict_dt, iris_real_dt, iris_Xtest_dt, iris_Ytest_dt, dt= IrisTrain_Decisiontree()
    cor_dt = IrisTrain_corr(iris_predict_dt,iris_real_dt)
    print("预测值与真实值得相关度：{:.2%}".format(cor_dt))
    plt.figure()
    plt.scatter(x_axis,iris_real_dt,color='red')
    plt.scatter(x_axis,iris_predict_dt,color='blue')
    plt.title("DT")
    plt.xlabel('the num of object')
    plt.ylabel('the value of feature')
    plt.show()

    iris_predict_rf, iris_real_rf, iris_Xtest_rf, iris_Ytest_rf, rf= IrisTrain_RandomForest()
    cor_rf = IrisTrain_corr(iris_predict_rf,iris_real_rf)
    print("预测值与真实值得相关度：{:.2%}".format(cor_rf))
    plt.figure()
    plt.scatter(x_axis,iris_real_rf,color='red')
    plt.scatter(x_axis,iris_predict_rf,color='blue')
    plt.title("RF")
    plt.xlabel('the num of object')
    plt.ylabel('the value of feature')
    plt.show()