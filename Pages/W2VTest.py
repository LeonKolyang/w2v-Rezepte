import streamlit as st
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import re
from Dataprocessing import w2v_gensim, KMeans as km
import gensim.models

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction


class Model_Test():
    def __init__(self):
        #Lade das Zutatenverzeichnis und das Kontrollframe zutatenDf
        self.zutaten_verzeichnis = pd.read_csv("Data/Doku_wordListNoAmount.csv", sep= "|", header=None)
        self.zutatenDf = pd.read_csv("Data/zutatenDf.csv", sep="|", header = 0)
        self.zutatenDf = self.zutatenDf.drop("Menge", axis=1)

    def body(self): 
        #dataset = st.sidebar.selectbox("Datensatz", ["Korpus mit Sonderzeichen", "Korpus ohne Sonderzeichen"])
        #if dataset == "Korpus mit Sonderzeichen":
        dataset = pd.read_csv("Data/Doku_corpusNoAmount.csv", sep= "|", header=None)

        no_iterations = st.sidebar.number_input("Anzahl Trainingsepochen", min_value=1, value= 5)
        window_size = st.sidebar.number_input("Wortfenstergröße", min_value=1, value=2)
        no_cluster = st.sidebar.number_input("Anzahl Cluster", min_value=1, value= 5)
        show_clusters=st.sidebar.checkbox("Zeige detaillierte Auswertung")
        results = None
        if st.button("Starte Testlauf"):
            results = self.run_test(dataset, no_iterations, no_cluster, window_size)
            st.text("Auswertung aller Cluster")
            st.dataframe(results[1])
            if show_clusters:
                st.text("Details der einzelnen Cluster")
                st.dataframe(results[0])
                    


    #Methode zum Aufrufen des Tests
    def run_test(self, dataset, no_iterations = 5, cluster_amount=5, window_size=2):
        input_parameters = pd.DataFrame(data=[no_iterations,window_size, cluster_amount],columns =["Parameter"], index=["Iterationen", "Fenstergröße", "Clusteranzahl"])
        st.dataframe(input_parameters)

        #Trainiere das Word2Vec Modell
        model_trainer = w2v_gensim.W2V()

        model_trainer.load_data(dataset)

        model_trainer.buildSentences()
        w2v=None
        with st.spinner("Modelltraining"):
            try:
                w2v=pd.read_csv("Data/gensim_w2v_"+str(no_iterations)+"_"+str(window_size)+".csv", header=0, sep="|", index_col=0)
            except :
                model_trainer.train_model(no_iterations,window_size)
                w2v = model_trainer.save_vectors(no_iterations, window_size)

        #Cluster die Vektoren aus dem Word2Vec Modell
        cluster_data = w2v[["x","y"]]
        kmeans = km.KMeans()
        with st.spinner("Clustering"):
            zuordnung = kmeans.run_manual_k(cluster_amount, cluster_data)
        
        w2v["Cluster"] = zuordnung["assigned to"]
        w2v.to_csv("Data/w2v_full_results_"+str(no_iterations)+"_"+str(window_size)+"_"+str(cluster_amount)+".csv", header=True, sep="|", index=True)

        #Erzeugen eines DataFrames zur Erzeugung der Kontrollergebnisse
        clusterlist = w2v["Cluster"].unique()
        results = pd.DataFrame(index=[clusterlist], columns=["Zugeordnete Wörter", "Daraus Bezeichnungen", "Reinheit"])
        for cluster in clusterlist:
            c_frame= w2v.loc[w2v["Cluster"]==cluster] 
            match_list = []
            for index, row in c_frame.iterrows():
                if row["labels"] in list(self.zutatenDf["Zuordnung"]):
                    match_list.append(row["labels"])
                
            results["Zugeordnete Wörter"][cluster] = len(c_frame)
            results["Daraus Bezeichnungen"][cluster] = len(match_list)
            results["Reinheit"][cluster] = round((len(match_list)/len(c_frame))*100,1)
        
        results = results.sort_index()
        new_indexes = []
        for index in list(results.index): 
            new_indexes.append("Cluster "+str(index[0]))
        results.index = new_indexes

        results.to_csv("Data/w2v_cluster_results_"+str(no_iterations)+"_"+str(window_size)+"_"+str(cluster_amount)+".csv", header=True, sep="|", index=True)

        #Ausgabe der Testläufe des Word2Vec Algorithmus
        result_index = ["Maximum", "Durchschnitt", "Minimum", "Über 50", "Unter 10"]
        result_data = [results["Reinheit"].max(), results["Reinheit"].mean(), results["Reinheit"].min(), 
                        len(results[results["Reinheit"] > 50]), len(results[results["Reinheit"] < 10])]
        result_zusammenfassung = pd.DataFrame(data=result_data, index=result_index, columns=["Reinheit"])
        return [results, result_zusammenfassung]

        # text = st.empty()
        # st.write(results)        
        # max = results["Hitrate"].max()
        # st.write("Max "+str(max))
        # avg = results["Hitrate"].mean()
        # st.write("Avg "+str(avg))
        # over50 = len(results[results["Hitrate"] > 50])
        # st.write(over50)
        # min = results["Hitrate"].min()
        # st.write("Min "+str(min))
        # under10 = len(results[results["Hitrate"] < 10])
        # st.write(under10)
            
        