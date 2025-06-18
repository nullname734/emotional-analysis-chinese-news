from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.db import connection
import jieba
import json
from collections import Counter
import joblib
import numpy as np
from keras._tf_keras.keras.models import load_model
from gensim.models import KeyedVectors
import os
import pickle
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

# 加载模型
w2v_model = KeyedVectors.load_word2vec_format('D:\\Users\\yang2\\Desktop\\毕业设计\\code\\training\\w2v.txt', binary=False)
svm_model = joblib.load('D:\\Users\\yang2\\Desktop\\毕业设计\\code\\training\\svm_w2v_v1.pk1')
lstm_model = load_model('D:\\Users\\yang2\\Desktop\\毕业设计\code\\training\\LSTM_model.h5')
stop_words = []
with open("D:\\Users\\yang2\\Desktop\\毕业设计\\code\\training\\hit_stopwords.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())
with open('D:\\Users\\yang2\\Desktop\\毕业设计\\code\\training\\label_encoder.pkl','rb') as f:
    encoder = pickle.load(f)

# 首页
def home(request):
    try:
        with connection.cursor() as cursor:
            # 获取总新闻量
            cursor.execute("SELECT COUNT(*) FROM today_news")
            total_news_count = cursor.fetchone()[0]
            
            # 获取今日新闻量（假设pub_date字段包含日期信息）
            cursor.execute("""
                select num from num where id = 1
            """)
            today_news_count = cursor.fetchone()[0]
            
        return render(request, 'home.html', {
            'total_news_count': total_news_count,
            'today_news_count': today_news_count
        })
    except Exception as e:
        print("获取新闻数量时发生错误：", str(e))
        return render(request, 'home.html', {
            'total_news_count': 0,
            'today_news_count': 0
        })


# 情感分析
def analyze_sentiment(titles, model_type):
    sentiment_counts = {'正面': 0, '中性': 0, '负面': 0}
    label_map = {'positive': '正面', 'neutral': '中性', 'negative': '负面'}
    for title in titles:
        words = jieba.lcut(title)
        vocabulary = set(w2v_model.index_to_key)
        if model_type == 'SVM':
            #计算平均词向量
            new_vectors = average_vector(words, w2v_model, vocabulary, 50)
            # 使用SVM模型进行预测
            predictions = svm_model.predict(new_vectors.reshape(1, -1))
            # 统计情感标签出现次数
            for pred in predictions:
                sentiment_counts[label_map[pred]] = sentiment_counts[label_map[pred]] + 1
        else:
            # 进行LSTM模型的预处理
            new_text = preprocessing(words, w2v_model)
            ls_predict = lstm_model.predict(new_text)
            predict_index = np.argmax(ls_predict, axis=1)
            predict_label = encoder.inverse_transform(predict_index)
            print(predict_label)
            for pred in predict_label:
                sentiment_counts[label_map[pred]] = sentiment_counts[label_map[pred]] + 1
    print(sentiment_counts)
    return sentiment_counts


# SVM模型需要计算平均词向量
def average_vector(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype='float64')
    nwords = 0
    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector += model[word]
    if nwords:
        feature_vector /= nwords
    return feature_vector

# LSTM的预处理是需要填充长度
def preprocessing(words, w2v, maxlen=300):
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

# 按照句子进行分析
def cut_content(news_content):
    # 按照句号分割文章
    sentences = news_content.split('。')
    # 去除空句子
    sentences = [sentence for sentence in sentences if sentence.strip()]
    new_sentences = []
    for sentence in sentences:
        if len(sentence) > 300:
            # lstm最长句子只能处理300, 所以需要重新截取
            for i in range(0, len(sentence), 300):
                new_sentences.append(sentence[i:i+300])
        else:
            new_sentences.append(sentence)
    return new_sentences

# 新闻列表
def news_list(request, model_type):
    try:
        # 查询获取新闻数据
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, content, the_source, ip,
                CASE 
                    WHEN pub_date IS NOT NULL THEN pub_date
                    ELSE '未知日期'
                END as pub_date
                FROM today_news 
                ORDER BY pub_date DESC
            """)
            columns = [col[0] for col in cursor.description]
            news_data = []
            titles = []  # 存储所有标题
            ip_counts = {}  # 存储IP地址计数
            date_counts = {}  # 存储日期计数
            
            for row in cursor.fetchall():
                news_dict = dict(zip(columns, row))
                news_data.append(news_dict)
                
                # 处理日期数据
                if news_dict['pub_date'] and news_dict['pub_date'] != '未知日期':
                    # 直接使用日期字符串，只取日期部分
                    date_str = news_dict['pub_date'].split()[0]  # 只取日期部分，去掉时间
                    date_counts[date_str] = date_counts.get(date_str, 0) + 1
                
                # /‘ip简称转换为完整的地名
                location_map = {
                    '北京': '北京市','上海': '上海市','天津': '天津市','重庆': '重庆市','河北': '河北省','山西': '山西省','辽宁': '辽宁省','吉林': '吉林省','黑龙江': '黑龙江省','江苏': '江苏省','浙江': '浙江省','安徽': '安徽省','福建': '福建省','江西': '江西省','山东': '山东省','河南': '河南省','湖北': '湖北省',
                    '湖南': '湖南省','广东': '广东省','海南': '海南省','四川': '四川省','贵州': '贵州省','云南': '云南省','陕西': '陕西省','甘肃': '甘肃省','青海': '青海省', '台湾': '台湾省',
                    '内蒙古': '内蒙古自治区','广西': '广西壮族自治区','西藏': '西藏自治区','宁夏': '宁夏回族自治区','新疆': '新疆维吾尔自治区','香港': '香港特别行政区','澳门': '澳门特别行政区'
                }
                if news_dict['ip']:
                    # 使用映射字典转换地名
                    location = news_dict['ip']
                    mapped_location = location_map.get(location, location)
                    ip_counts[mapped_location] = ip_counts.get(mapped_location, 0) + 1
                    
        #分页，每页20条新闻数据
        try:
            page = int(request.GET.get('page', 1))
        except:
            page = 1

        per_page = 20
        start = (page - 1) * per_page
        end = start + per_page
        paginated_data = news_data[start:end]
        total_pages = (len(news_data) + per_page - 1) // per_page
        titles = (item['title'] for item in paginated_data)

        # 进行这一页新闻标题的情感分析
        sentiment_counts = analyze_sentiment(titles, model_type)
        
        # 准备地图热力图数据
        map_data = []
        for location, count in ip_counts.items():
            map_data.append({
                'name': location,
                'value': count
            })
            
        # 准备日历图数据
        news_dates = [[date, count] for date, count in date_counts.items()]
        
        #返回数据到前端
        return render(request, 'news_list.html', {
            'news_list': paginated_data,
            'current_page': page,
            'total_pages': total_pages,
            'model_type': model_type,
            'sentiment_data': json.dumps(sentiment_counts),
            'map_data': json.dumps(map_data, ensure_ascii=False),
            'news_dates': json.dumps(news_dates, ensure_ascii=False)
        })
    except Exception as e:
        print("发生错误：", str(e))
        return render(request, 'error.html', {'error': str(e)})


def news_detail(request, news_id, model_type):
    try:
        # 查询获取单条新闻详情
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, content, the_source, 
                CASE 
                    WHEN pub_date IS NOT NULL THEN pub_date
                    ELSE '未知日期'
                END as pub_date
                FROM today_news 
                WHERE id = %s
            """, [news_id])
            columns = [col[0] for col in cursor.description]
            news = dict(zip(columns, cursor.fetchone()))

            # 生成词云数据
            content = news['content']
            # 使用jieba进行分词
            words = jieba.cut(content)
            # 过滤停用词
            words = [word for word in words if len(word) > 1 and word not in stop_words]
            # 统计词频
            word_freq = Counter(words)
            # 转换为词云需要的格式
            word_frequencies = [{'name': word, 'value': freq} for word, freq in word_freq.most_common(50)]

            # 生成关系网络图数据
            top_words = [word for word, _ in word_freq.most_common(20)]
            # 计算词之间的相似度
            relation_data = []
            for i, word1 in enumerate(top_words):
                for j, word2 in enumerate(top_words[i+1:], i+1):
                    if word1 in w2v_model and word2 in w2v_model:
                        similarity = float(w2v_model.similarity(word1, word2))  #float类型
                        if similarity > 0.5:  #大于0.5的关系
                            relation_data.append({
                                'source': word1,
                                'target': word2,
                                'value': similarity
                            })

        #print("处理后的新闻详情：", news)
        #print("词云数据：", word_frequencies)

        # 情感分析新闻内容
        split_news = cut_content(news['content'])
        sentiment_counts = analyze_sentiment(split_news, model_type)
        return render(request, 'news_detail.html', {
            'news': news,
            'word_frequencies': json.dumps(word_frequencies, ensure_ascii=False),
            'model_type': model_type,
            'sentiment_data': json.dumps(sentiment_counts),
            'relation_data': json.dumps(relation_data, ensure_ascii=False)
        })
    except Exception as e:
        print("发生错误：", str(e))
        return render(request, 'error.html', {'error': str(e)})