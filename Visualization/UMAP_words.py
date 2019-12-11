import pickle
import os.path
import numpy as np
import pandas as pd
import umap


if __name__ == '__main__':
    path = os.getcwd()
    with open('./Pickles/word2vec_4.pkl', 'rb') as file:
        w2v_model = pickle.load(file)
    words = set(w2v_model.wv.vocab)
    vectors = []
    vocab = []
    for word in words:
        vectors.append(w2v_model.wv.__getitem__([word]))
        vocab.append(word)
    vectors = np.asarray(vectors)
    vectors = vectors.reshape(-1, 300)
    umapper = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2, random_state=42)
    word_mapper = umapper.fit(vectors)
    word_map = word_mapper.transform(vectors)
    w2v = pd.DataFrame({'x': [x for x in word_map[:, 0]],
                        'y': [y for y in word_map[:, 1]],
                        'word': vocab})
    with open(path + '/Visualization/umap_words.pkl', 'wb') as file:
        pickle.dump([w2v, word_mapper], file)
