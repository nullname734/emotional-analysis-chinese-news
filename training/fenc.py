import jieba
import numpy as np
import pandas as pd
import sqlite3
import jieba.posseg as pseg
from transformers import pipeline
from paddlenlp import Taskflow

#连接数据库
con = sqlite3.connect('D:\\Users\\yang2\\Desktop\\毕业设计\\code\\sc_news\\news.db')
cursor = con.cursor()

#存入数据库的表
cursor.execute('''
CREATE TABLE IF NOT EXISTS content (
    id integer primary key autoincrement,
    content TEXT,
    word TEXT,
    emotion TEXT,
    score double
)
''')

#读取数据
query = "SELECT id, 内容 FROM news"
data = pd.read_sql_query(query, con)

#print(data.head())

#重新过滤换行符\n，防止漏网之鱼
data["内容"] = data["内容"].apply(lambda x: x.replace("\n", ""))
#new_data = data
#print(data.head())

#加载停用词表
stop_words = []
with open("hit_stopwords.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())


#是否中文
def is_chinese(strs):
    for str in strs:
        if not '\u4e00' <= str <= '\u9fa5':
            return False
    return True


#进行分词并标注词性（名词、动词、形容词）
def word_cut(content):
    #分词并标注词性
    words = pseg.cut(content)
    noun_adj = []
    #是中文且不是停用词的就切割并保存
    for word, flag in words:
        if word not in stop_words and len(word) >= 1 and is_chinese(word) == True:
            noun_adj.append(word)
    if len(noun_adj) != 0:
        return ' '.join(noun_adj)
    else:
        return np.NAN


#为了标注，将新闻语句按照句子进行分割。
def split(data, column='内容'):
    #分割指定列的数据
    split_data = data[column].str.split('。', expand=True).stack()

    split_data = split_data.reset_index(level=1, drop=True).reset_index(name=column)

    split_data = split_data[split_data[column] != '']

    split_data = split_data.reset_index(drop=True)

    return split_data


def split_long(row):
    content = row['内容']
    chunks = []
    start = 0
    #分割
    while start < len(content):
        chunks.append(content[start:start + 512])
        start += 512
    return chunks


new_data = split(data)
#new_data['内容'] = new_data['内容'].apply(lambda x: [x[i:i+512] for i in range(0, len(x),512)])
#new_data = new_data.explode('内容').reset_index(drop=True)
#分完后对于超出512个字的数据依然需要再次分割

#new_data[['内容', 'word']].to_sql('content', con, if_exists='replace', index=False)
#标注情感
# model_path = "D:\\Users\\yang2\\Desktop\\毕业设计\\roberta-base-finetuned-jd-binary-chinese"
# classifier = pipeline('sentiment-analysis', model=model_path, device='cuda')
# label_list = []
# score_list = []
# for content in new_data['内容']:
#     class1 = classifier(content)
#     label = class1[0]['label']
#     score = class1[0]['score']
#     if score <= 0.96:
#         label = 'neutral'
#         label_list.append(label)
#     else:
#         label = label
#         label_list.append(label)
#     score_list.append(score)


#情感分类标注
label_list = []
score_list = []
classifier = Taskflow("sentiment_analysis", device_id='cuda')
for content in new_data['内容']:
    c = classifier(content)
    label = c[0]['label']
    score = c[0]['score']
    if score <= 0.8:
        label = 'neutral'
        label_list.append(label)
    else:
        label = label
        label_list.append(label)
    score_list.append(score)
# 保存到数据库

new_data['emotions'] = label_list
new_data['score'] = score_list
new_data['word'] = new_data['内容'].apply(word_cut)
new_data.to_sql('content', con, if_exists='replace', index=False)
#提交更改，关闭连接
con.commit()
con.close()
