from gensim.models import word2vec


raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep","可以 识别 中文 吗"]

# 切分词汇
sentences= [s.split() for s in raw_sentences]

print(sentences[0][3])

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=1)

y = -9
t = model.wv.most_similar(positive=["fox", "dogs"], negative=["you"])
y1 = model.wv["dex"]
print(t)
y = 7