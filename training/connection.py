import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import jieba

model = KeyedVectors.load_word2vec_format('w2v.txt', binary=False)

text = '自然语言处理于机器学习是人工智能的重要组成部分。'
text = ' '.join(jieba.cut(text))
words = [word.lower() for word in text.split() if word.lower() in model]
print('有效词汇', words)

vectors = np.array([model[word] for word in words])
similarity_matrix = cosine_similarity(vectors)

G = nx.Graph()
threashold = 0.5

for word in words:
    G.add_node(word)

for i in range(len(words)):
    for j in range(i+1, len(words)):
        similarity = similarity_matrix[i][j]
        if similarity > threashold:
            G.add_edge(words[i], words[j], weight=similarity)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5, seed=42)
# 绘制节点和边
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')
nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, alpha=0.7)

# 添加标签
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# 添加边权重标签
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# 图表设置
plt.title("Text Semantic Network (GloVe-based Similarity)", size=15)
plt.axis('off')

# 显示图表
plt.show()