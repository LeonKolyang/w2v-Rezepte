import streamlit as st
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import re
import w2v_gensim
import gensim.models
import KMeans as km

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction


class Model_Test():
    def __init__(self):
        #Lade das Zutatenverzeichnis und das Kontrollframe zutatenDf
        self.zutaten_verzeichnis = pd.read_csv("Data/Doku_wordListNoAmount.csv", sep= "|", header=None)
        self.zutatenDf = pd.read_csv("Data/zutatenDf.csv", sep="|", header = 0)
        self.zutatenDf = self.zutatenDf.drop("Menge", axis=1)

    def body(self):
        no_iterations = st.number_input("Anzahl Trainingsepochen", min_value=1, value= 5)
        no_cluster = st.number_input("Anzahl Cluster", min_value=1, value= 5)

        if st.button("Starte Testlauf"):
            self.run_test(no_iterations, no_cluster)


    #Methode zum Aufrufen des Tests
    def run_test(self, no_iterations = 5, cluster_amount=5):
        
        #Trainiere das Word2Vec Modell
        model_trainer = w2v_gensim.W2V()

        model_trainer.load_data(self.zutaten_verzeichnis)
        text = st.write("Modelltraining gestartet mit Zutatenverzeichnis")

        model_trainer.buildSentences()
        model_trainer.train_model(no_iterations)
        w2v = model_trainer.save_vectors()

        #Cluster die Vektoren aus dem Word2Vec Modell
        text = st.write("Training beendet, starte Clustering")
        cluster_data = w2v[["x","y"]]
        kmeans = km.KMeans()
        zuordnung = kmeans.run_manual_k(cluster_amount, cluster_data)
        
        w2v["Cluster"] = zuordnung["assigned to"]

        #Erzeugen eines DataFrames zur Erzeugung der Kontrollergebnisse
        clusterlist = w2v["Cluster"].unique()
        results = pd.DataFrame(index=[clusterlist], columns=["Zugeordnet", "Matches", "Hitrate"])
        for cluster in clusterlist:
            c_frame= w2v.loc[w2v["Cluster"]==cluster]
            match_list = []
            for index, row in c_frame.iterrows():
                if row["labels"] in list(self.zutatenDf["Zuordnung"]):
                    match_list.append(row["labels"])
                
            results["Zugeordnet"][cluster] = len(c_frame)
            results["Matches"][cluster] = len(match_list)
            results["Hitrate"][cluster] = round((len(match_list)/len(c_frame))*100,1)

        #Ausgabe der Testläufe des Word2Vec Algorithmus
        text = st.empty()
        st.write(results)        
        max = results["Hitrate"].max()
        st.write("Max "+str(max))
        avg = results["Hitrate"].mean()
        st.write("Avg "+str(avg))
        over50 = len(results[results["Hitrate"] > 50])
        st.write(over50)
        min = results["Hitrate"].min()
        st.write("Min "+str(min))
        under10 = len(results[results["Hitrate"] < 10])
        st.write(under10)
            
        