import tensorflow as tf
import jieba
import numpy as np
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model
import pickle
from gensim.models import KeyedVectors

model = load_model('lstm_model.h5')
w2v_file = "w2v.txt"
w2v_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
with open('label_encoder.pkl','rb') as f:
    encoder = pickle.load(f)


def preprocessing(text, w2v, maxlen=300):
    words = jieba.lcut(text)
    #转换词向量
    embedded=[]
    for word in words:
        #模型有的词转换成词向量，没有的填充零
        if word in w2v:
            embedded.append(w2v[word])
        else:
            embedded.append(np.zeros(w2v.vector_size))
    padded = pad_sequences(
        [embedded], dtype='float32', padding='post', maxlen=maxlen
    )
    return padded

Text = '我今天心情很好'
new_text = preprocessing(Text, w2v_model)
predict = model.predict(new_text)
predict_index = np.argmax(predict, axis=1)
predict_label = encoder.inverse_transform(predict_index)
print(predict_label)