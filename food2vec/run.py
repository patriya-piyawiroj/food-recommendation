model = gensim.models.Word2Vec.load("food2vec_recipe1M.model")
print(model.wv.most_similar(positive=["nacho chips", "salsa verde"], topn=20))