from sklearn import datasets #数据来源
from sklearn.model_selection import train_test_split #数据分割
from sklearn import svm #模型选择
from sklearn.neighbors import KNeighborsClassifier #模型选择
from sklearn.neural_network import MLPClassifier #模型选择
from sklearn.naive_bayes import GaussianNB #模型选择
from sklearn import tree
from sklearn.linear_model import SGDClassifier #
from sklearn.ensemble import RandomForestClassifier
import joblib #保存模型

test_num = 45   #全局变量 测试样本数

def IrisTrain_svm():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    s = svm.SVC(kernel='linear')
    s.fit(x_train,y_train)
    iris_predict = s.predict(x_test)  # 预测
    return iris_predict,x_test,y_test,s

def IrisTrain_knn():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    knn = KNeighborsClassifier(n_neighbors=1)    #选择模型
    knn.fit(x_train,y_train)    #训练
    iris_predict = knn.predict(x_test)     #预测
    return iris_predict,x_test,y_test,s

def IrisTrain_mlp():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    mlp = MLPClassifier(max_iter=1000)    #选择模型
    mlp.fit(x_train,y_train)    #训练
    iris_predict = mlp.predict(x_test)     #预测
    return iris_predict,x_test,y_test,mlp

def IrisTrain_gnb():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    gnb = GaussianNB()    #选择模型
    gnb.fit(x_train,y_train)    #训练
    iris_predict = gnb.predict(x_test)     #预测
    return iris_predict,x_test,y_test,gnb

def IrisTrain_Decisiontree():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    dt = tree.DecisionTreeClassifier()    #选择模型
    dt.fit(x_train,y_train)    #训练
    iris_predict = dt.predict(x_test)     #预测
    return iris_predict,x_test,y_test,dt

def IrisTrain_sgd():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    sgd = SGDClassifier()    #选择模型
    sgd.fit(x_train,y_train)    #训练
    iris_predict = sgd.predict(x_test)     #预测
    return iris_predict,x_test,y_test,sgd

def IrisTrain_RandomForest():
    iris_data = datasets.load_iris()    #加载
    x_train = iris_data.data   #特征集
    y_train = iris_data.target     #标签
    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.3)     #切割训练集和测试集
    rf = RandomForestClassifier()    #选择模型
    rf.fit(x_train,y_train)    #训练
    iris_predict = rf.predict(x_test)     #预测
    return iris_predict,x_test,y_test,rf

def Iris_score(iris_Ytest,iris_predict):        #判断测试集的准确率
    corret_num = 0
    for i in range(len(iris_predict)):
        if iris_Ytest[i] == iris_predict[i]:
            corret_num = corret_num + 1
    return corret_num/test_num

if __name__ == '__main__':
    lables = ["德国鸢尾", "荷兰鸢尾", "网脉鸢尾"]

    iris_predict_s, iris_Xtest_s, iris_Ytest_s, s= IrisTrain_svm()
    # for i in range(len(iris_predict_s)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest_s[i]], lables[iris_predict_s[i]]))
    print("svm的准确率是：{:.2%}".format(s.score(iris_Xtest_s, iris_Ytest_s)))

    iris_predict_knn, iris_Xtest_knn, iris_Ytest_knn,knn= IrisTrain_knn()
    # for i in range(len(iris_predict)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest[i]], lables[iris_predict[i]]))
    #     if iris_Ytest[i] == iris_predict[i]:
    #         num = num + 1
    #     else:
    #         print(i)
    print("knn的准确率是：{:.2%}".format(knn.score(iris_Xtest_knn, iris_Ytest_knn)))
    print("knn的测试集准确率是：{:.2%}".format(Iris_score(iris_Ytest_knn, iris_predict_knn)))
    # print(float(Iris_score(iris_Ytest,iris_predict)))

    iris_predict_mlp, iris_Xtest_mlp, iris_Ytest_mlp, mlp= IrisTrain_mlp()
    # for i in range(len(iris_predict)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest[i]], lables[iris_predict[i]]))
    print("mlp的准确率是：{:.2%}".format(mlp.score(iris_Xtest_mlp, iris_Ytest_mlp)))
    print("mlp的测试集准确率是：{:.2%}".format(Iris_score(iris_Ytest_mlp, iris_predict_mlp)))
    #
    iris_predict_gnb, iris_Xtest_gnb, iris_Ytest_gnb, gnb= IrisTrain_gnb()
    # for i in range(len(iris_predict)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest[i]], lables[iris_predict[i]]))
    print("gnb的准确率是：{:.2%}".format(gnb.score(iris_Xtest_gnb, iris_Ytest_gnb)))
    print("gnb的测试集准确率是：{:.2%}".format(Iris_score(iris_Ytest_gnb, iris_predict_gnb)))

    iris_predict_dt, iris_Xtest_dt, iris_Ytest_dt, dt= IrisTrain_Decisiontree()
    # for i in range(len(iris_predict)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest[i]], lables[iris_predict[i]]))
    print("dt的准确率是：{:.2%}".format(dt.score(iris_Xtest_dt, iris_Ytest_dt)))
    print("dt的测试准确率是：{:.2%}".format(Iris_score(iris_Ytest_dt, iris_predict_dt)))

    iris_predict_sgd, iris_Xtest_sgd, iris_Ytest_sgd, sgd= IrisTrain_sgd()
    # for i in range(len(iris_predict)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest[i]], lables[iris_predict[i]]))
    print("sgd的准确率是：{:.2%}".format(sgd.score(iris_Xtest_sgd, iris_Ytest_sgd)))
    print("sgd的测试准确率是：{:.2%}".format(Iris_score(iris_Ytest_sgd, iris_predict_sgd)))

    iris_predict_rf, iris_Xtest_rf, iris_Ytest_rf, rf= IrisTrain_RandomForest()
    # for i in range(len(iris_predict)):
    #     print("第%s次测试，预测值是：%s，真实值是：%s" % ((i + 1), lables[iris_Ytest[i]], lables[iris_predict[i]]))
    print("rf的准确率是：{:.2%}".format(rf.score(iris_Xtest_rf, iris_Ytest_rf)))
    print("rf的测试准确率是：{:.2%}".format(Iris_score(iris_Ytest_rf, iris_predict_rf)))