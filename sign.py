
from transformers import pipeline
import sqlite3
import pandas as pd
import torch

#使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
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
query = "SELECT 内容,word FROM content"
data = pd.read_sql_query(query, con)

#调用现成的情感分类模型，标注新闻文本
model_path = "D:\\Users\\yang2\\Desktop\\毕业设计\\roberta-base-finetuned-jd-binary-chinese"
classifier = pipeline('sentiment-analysis', model=model_path, device=device)
label_list = []
score_list = []
for content in data['内容']:
    class1 = classifier(content)
    label = class1[0]['label']
    score = class1[0]['score']
    if score <= 0.6:
        label = 'NEUTRAL'
        label_list.append(label)
    else:
        label = label
        label_list.append(label)
    score_list.append(score)
print(label_list)
print(score_list)
data['emotions'] = label_list
data['score'] = score_list
data.to_sql('content', con, if_exists='replace', index=False)
#提交更改，关闭连接
con.commit()
con.close()