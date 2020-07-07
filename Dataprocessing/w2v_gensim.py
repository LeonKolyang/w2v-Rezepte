import gensim.models
import pandas as pd
import streamlit as st

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling
import matplotlib.pyplot as plt

class W2V():
    def __init__(self):
        self.zutaten_verzeichnis = None
        self.sentences = []
        self.model = None

    #Lade das Zutatenverzeichnis auf dem trainiert wird
    def load_data(self, data):
        self.zutaten_verzeichnis = data

    #Baue aus dem Zutatenverzeichnis die Sätze
    #@st.cache
    def buildSentences(self):
        for index, row in self.zutaten_verzeichnis.iterrows():
            list = row[0].split(" ")
            if len(list[-1]) == 0:
                list = list[:-1]
            self.sentences.append(list)
        #Testausgabe der Sätze
        #for i, sentence in enumerate(self.sentences):
        #    if i == 10:
        #        break
        #    print(sentence)

    #@st.cache
    def train_model(self, no_iterations, window_size):
        #Aufruf des Trainings desWord2Vec Algorithmus mit den in der Arbeit beschriebenen Parametern
        self.model = gensim.models.Word2Vec(self.sentences, sg=1,min_count=0, size= 300, negative=5, iter=no_iterations, window=window_size)


    #gensim implementierung des scikit-learn Dimensionsreduzieren
    #@st.cache
    def reduce_dimensions(self, model):
        num_dimensions = 2  # final num dimensions (2D, 3D, etc)

        vectors = [] # positions in vector space
        labels = [] # keep track of words to label our data again later
        for word in model.wv.vocab:
            vectors.append(model.wv[word])
            labels.append(word)

        # convert both lists into numpy vectors for reduction
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)

        # reduce using t-SNE
        vectors = np.asarray(vectors)
        tsne = TSNE(n_components=num_dimensions)
        vectors = tsne.fit_transform(vectors)

        x_vals = [v[0] for v in vectors]
        y_vals = [v[1] for v in vectors]
        return x_vals, y_vals, labels

    def save_vectors(self, no_iterations, window_size):
        #Reduzierung der Dimensionen
        x_vals, y_vals, labels = self.reduce_dimensions(self.model)

        #Speichere die finalen Wordembeddings als .csv Datei ab
        values = pd.DataFrame({"x":x_vals,"y" :y_vals,"labels":labels})
        values.to_csv("../Data/gensim_w2v_"+str(no_iterations)+"_"+str(window_size)+".csv", header=True, sep="|")
        return values

