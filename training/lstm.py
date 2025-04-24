import sqlite3
import pandas as pd
import sklearn.model_selection as ms
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras._tf_keras.keras import models, layers
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import RMSprop, Adam, SGD
import keras._tf_keras.keras as tf_k
# from  tensorflow.keras import models, layers
# from  tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import RMSprop, Adam, SGD
# import tensorflow.keras as tf_k
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau

#数据集加载
def load_data(path, query):
    con = sqlite3.connect(path)
    cursor = con.cursor()
    data = pd.read_sql_query(query, con)
    return data


#转换嵌入空间中的单词矩阵
def embed_setence(w2v_model, sentence):
    content = []
    sentence = sentence.split()
    #print(sentence)
    for word in sentence:
        #print(word)
        if word in w2v_model:
            content.append(w2v_model[word])
        else:
            content.append(np.zeros(w2v_model.vector_size))
    return np.array(content)


#将句子列表转换为矩阵列表
def embeding(w2v_model, sentences):
    embed = []
    for sentence in sentences:
        embed_s = embed_setence(w2v_model, sentence)
        embed.append(embed_s)
    return embed


#对于lstm来说，得到的数据需要长度相同，短了的为矩阵补0
def padding(w2v_model, X_data, maxlen):
    #求取向量
    X_embed = embeding(w2v_model, X_data)
    #对短了的值补零
    X_pad = pad_sequences(X_embed, dtype='float32', padding='post', maxlen=maxlen)
    return X_pad


#LSTM建模
def lstm(train_d, w2v_model):
    X = train_d['word']
    Y = train_d['emotions']

    
    #原有的Y标签均为文字，需要转换成数字
    le = LabelEncoder()
    y = le.fit_transform(Y).astype('float32')
    #保存LabelEncoder供后续使用
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    # 训练时的标签映射
    # label_dict = {"negative": 0.0, "neutral": 1.0, "positive": 2.0}
    # # 构建反向映射字典
    # reverse_label_dict = {v: k for k, v in label_dict.items()}
    # # 将预测数值转换为文本
    # y = [reverse_label_dict[pred] for pred in Y]
    x_train, x_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=7)

    print('正在嵌入词模型')
    #嵌入词向量
    X_train_pad = padding(w2v_model, x_train, maxlen=300)
    X_test_pad = padding(w2v_model, x_test, maxlen=300)

    #RNN
    print('正在构建模型')
    model = models.Sequential()  #线性堆叠网络
    model.add(layers.Masking(mask_value=0, input_shape=(300, 50)))  #masking层
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                64,
                activation='tanh',
                dropout=0.5,
                recurrent_dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            )
        ))  #双向LSTM层
    #批标准化层
    model.add(layers.BatchNormalization())
    #全连接层
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(20, activation='relu'))
    #输出神经元
    model.add(layers.Dense(3, activation='softmax'))

    #编译模型
    print('编译模型中……')
    model.compile(
        loss='sparse_categorical_crossentropy',  #根据标签类型选择损失函数
        optimizer=Adam(learning_rate=0.00005),
        metrics=['accuracy']
    )

    #可视化网络结构
    print('绘制模型结构')
    tf_k.utils.plot_model(
        model,
        to_file='LSTM_model.png',  #保存模型图像
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=200
    )

    #早停法
    es = EarlyStopping(
        monitor='val_accuracy',  #监控验证准确率
        patience=6, #允许连续10轮指标不改善
        restore_best_weights=True,  #恢复最佳模型权重
        verbose=1
    )

    #动态学习调度,尽量收敛
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )

    #训练
    print('正在训练模型')
    train = model.fit(
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        verbose=1, #简短训练日志
        callbacks=[es, reduce_lr]
    )

    #预测
    print('正在测试模型……')
    predict = model.predict(X_test_pad)
    #评估时需要恢复分类值
    y_pred = np.argmax(predict, axis=1)
    y_pred_text = le.inverse_transform(y_pred)
    y_test_int = y_test.astype('int')
    y_test_text = le.inverse_transform(y_test_int)
    score = accuracy_score(y_test_text, y_pred_text)
    rep = classification_report(y_pred_text, y_test_text)
    res = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f'The accuracy evaluated on the test is of {res[1] * 100:.3f}%')

    #绘制训练曲线
    print('正在绘制训练曲线……')
    plot_training_history(train)
    plt.show()

    #保存训练结果
    print('正在保存模型……')
    model.save('LSTM_model.h5')
    print('模型保存成功')

    return score, rep

def plot_training_history(history):
    # 绘制拟合曲线，是否过拟合
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train_accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_title('accuracy')
    ax2.legend()

if __name__ == '__main__':
    # 读取数据
    print("读取数据集")
    data_path = 'D:\\Users\\yang2\\Desktop\\毕业设计\\code\\sc_news\\news.db'
    # 使用已经预处理后的数据
    query = "SELECT * FROM data"
    data = load_data(data_path, query)
    #导入gloVe模型
    print("加载词向量模型")
    w2v_file = "w2v.txt"
    g_model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

    #训练模型
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用显存动态增长
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # 显式初始化 GPU 上下文
            with tf.device('/GPU:0'):
                dummy_tensor = tf.constant(1)
                print("GPU 初始化成功，设备:", dummy_tensor.device)
        except RuntimeError as e:
            print("GPU 初始化失败:", e)
    else:
        print("未检测到 GPU 设备。")

    print("训练lstm模型")
    acc, report = lstm(data, g_model)
    print(f'训练准确度为：{acc}')
    print(report)