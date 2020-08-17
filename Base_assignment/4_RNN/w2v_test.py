from gensim.models import Word2Vec
word2vec_model = Word2Vec.load("w2v_all.model")
# 查看词向量
print('hello：', word2vec_model['hello'])
# 查看相似词
sim_words = word2vec_model.most_similar('hello')
for w in sim_words:
    print(w)
