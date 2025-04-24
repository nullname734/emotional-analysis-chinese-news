import joblib
import numpy as np
from gensim.models import KeyedVectors
import re
import jieba

def clean_and_segment(text):
    clean = re.sub(r'[^\w\s]]','',text)
    words = jieba.lcut(clean)
    return words

def average_vector(words,model,vocabulary,num_features):
    feature_vector = np.zeros((num_features,),dtype='float64')
    nwords = 0
    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector += model[word]
    if nwords:
        feature_vector /= nwords
    return feature_vector


if __name__ == '__main__':
    # 加载模型
    svm_model = joblib.load('svm_w2v_v1.pk1')
    w2v_file = "w2v.txt"
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

    # 动态获取词向量维度
    num = w2v_model.vector_size
    test_S =['亏了168亿，彻底破产，崩溃了']
    procees = [clean_and_segment(text) for text in test_S]

    #转换词向量
    vocabulary = set(w2v_model.index_to_key)
    new_vectors = np.array([
        average_vector(text, w2v_model, vocabulary, num_features=50)
        for text in procees
    ])

    #预测类别标签
    predictions = svm_model.predict(new_vectors)
    label_map = {'positive':'正面', 'neutral':'中性','negative':'负面'}
    print(predictions)
    for text, pred in zip(test_S, predictions):
        print(f'文本：{text}\n预测类别：{label_map[pred]}\n')