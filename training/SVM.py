import joblib
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import sklearn.model_selection as ms
import sqlite3
import keras._tf_keras.keras as tf_k

#数据集加载
def load_data(path, query):
    con = sqlite3.connect(path)
    cursor = con.cursor()
    data = pd.read_sql_query(query, con)
    return data


#将分词的内容进行向量化
def vectorize(data, w2v):
    vectorized_text = w2v.tranform([data['word']])
    #需要输出2D的特征数量
    if vectorized_text.shape[0] == 0:
        return np.zeros((1, w2v.dim))
    return vectorized_text


#平均词向量
def average_vectors(words, model, vocabulary, num_features):
    #num_features词向量的维度，gloVe为300
    feature_vec = np.zeros((num_features,), dtype='float64')
    nwords = 0
    for word in words:
        #检查单词在词向量字典中是否存在
        if word in vocabulary:
            nwords = nwords + 1
            feature_vec = feature_vec + model[word]
    #计算平均值
    if nwords:
        feature_vec /= nwords
    return feature_vec


#文本转换为平均词向量
def word_average_vector(corpus, model, num_features):
    #模型的词典
    vocabulary = set(model.index_to_key)
    feature = [average_vectors(sentence, model, vocabulary, num_features)
               for sentence in corpus]
    return np.array(feature)


#获取数据词向量
def get_word_vec(data, model):
    #将空格进行分离
    words = [text.split(' ') for text in data]
    return word_average_vector(words, model, num_features=50)


#训练SVM模型
def svm_model(train_d, w2v_model):
    X = train_d['word']
    Y = train_d['emotions']
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=7)
    #print(x_train)

    #训练和测试数据的词向量,向量需要嵌入svm
    x_train_w2v = get_word_vec(x_train, w2v_model)
    x_test_w2v = get_word_vec(x_test, w2v_model)

    #使用网格搜索调参
    param_grid = {
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-3, 3, 7)
    }  #拓展参数网络

    #最佳参数：C=10.0, gamma=1.0(调整模型使用）

    #网格搜索
    # print('寻找最佳参数')
    # grid_search = ms.GridSearchCV(
    #     SVC(kernel='rbf'),
    #     param_grid,
    #     cv=5,
    #     scoring='f1_weighted',
    #     n_jobs=-1,
    #     verbose=2
    # )
    # grid_search.fit(x_train_w2v, y_train)

    #输出最佳参数
    # best_params = grid_search.best_params_
    # print(f"最佳参数：C={best_params['C']}, gamma={best_params['gamma']}")

    #训练模型
    # final_model = grid_search.best_estimator_
    params = {
        'C': 10.0,
        'kernel': 'rbf',
        'gamma': 1.0
    }
    final_model = SVC(**params)
    final_model.fit(x_train_w2v, y_train)

    #评估模型
    predict = final_model.predict(x_test_w2v)
    accuracy = accuracy_score(y_test, predict)
    report = classification_report(y_test, predict, target_names=['positive', 'neutral', 'negative'])

    #保存模型
    model_path = f'svm_w2v_v1.pk1'
    joblib.dump(final_model, model_path)
    return accuracy, report


if __name__ == '__main__':
    #读取数据
    print("读取数据集")
    data_path = 'D:\\Users\\yang2\\Desktop\\毕业设计\\code\\sc_news\\news.db'
    #使用已经预处理后的数据
    query = "SELECT * FROM data"
    data = load_data(data_path, query)
    #读取成功后进行训练
    if data is not None:
        print("加载词向量模型")
        w2v_file = "w2v.txt"
        g_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

        print("进行SVM模型的构建")
        acc, report = svm_model(data, g_model)
        print(f'测试准确率：{acc:.2f}')
        print(report)
