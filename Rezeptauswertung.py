import sys
import streamlit as st
import pandas as pd
from pymongo import MongoClient
#from google.cloud import pubsub_v1
#from google.cloud import firestore
#import pubsub as pb
import requests
import csv
import time
import json
import datetime


from Pages import *
#import Concept
#import Vectorization
#import KMeans
from PIL import Image


import matplotlib.pyplot as plt


class Sidebar:
    def __init__(self):
        pass

    def navigator_with_tool(self):
        st.sidebar.header("Data Grabbing und Preprocessing für Word2Vec")
        routes = ["Tool Test", "Projektvorstellung", "Data Grabbing", "Data Preprocessing", "Modell Test", "Modellauswertung"]

        return st.sidebar.radio("Go to", routes)
    
    def navigator_without_tool(self):
        st.sidebar.header("Data Grabbing und Preprocessing für Word2Vec")
        routes = ["Projektvorstellung", "Data Grabbing", "Data Preprocessing", "Modell Test", "Modellauswertung"]

        return st.sidebar.radio("Go to", routes)
        
    def dataSelector(self):
        return st.sidebar.selectbox("Datensatz",("Vollständiger Datensatz", "Test Datensatz"),index=0)

    def dataReturn(self, inputFile):
        if inputFile == "Vollständiger Datensatz":
            return self.DP.dataCaller("Data/Doku_reweIngredients.csv")
        elif inputFile == "Test Datensatz":
            return self.DP.dataCaller("Data/Doku_reweIngredientsSmall.csv")

    def reloadButton(self):
        return st.sidebar.button("Reload Data", key="r1")

    def saveData(self):
        return st.sidebar.button("Preprocessing Daten speichern", key="dSave")

    def saver(self, data):
        wordSeries = pd.Series(data["wordList"])
        wordSeries.to_csv('Data/Doku_wordList.csv',header=False, index=False,sep="§")
        corpusSeries = pd.Series(data["corpus"])
        corpusSeries.to_csv('Data/Doku_corpus.csv',header=False, index=False,sep="§")


    def openData(self):
        return st.sidebar.button("Load Data for Machine Learning", key="dOpen")

    def opener(self):
        dataDict = {    "wordList": pd.read_csv('Data/Doku_wordList.csv', sep ="§", encoding = "UTF-8", header = None, engine="python"),
                        "corpus":   pd.read_csv('Data/Doku_corpus.csv',sep="§", encoding = "UTF-8", header = None, engine="python")}
        return dataDict



def main():
    pipeline = Pipeline.Pipeline()
    pipeline_connection = False
    try: 
        pipeline.start_service()
        pipeline_connection = True
    except Exception as e:
        pipeline_connection = False
        sys.stdout.write(str(e))


    title = Title.Title()
    finaldata = Finaldata.finalData()
    grabbing = Grabbing.Grabbing()
    preprocessing = Preprocessing.Preprocessing()
    w2vTest = W2VTest.Model_Test()
    auswertung = Auswertung.Auswertung()
    
    sidebar = Sidebar()
    visualiser = st.empty()
    saver = False

    try:
        MLdata = sidebar.opener()
    except:
        st.write("No Data available. Save Data first.")

    if pipeline_connection:
        tab = sidebar.navigator_with_tool()
    else:
        tab = sidebar.navigator_without_tool()
    dataSelector = st.empty()
    data = None
    text = st.sidebar.empty()    

    if tab == "Tool Test":
        visualiser.empty()  
        visualiser = pipeline.body()

    if tab == "Projektvorstellung":
        st.header("Data Grabbing und Preprocessing für Word2Vec")
        st.write("Vorbereitung einer Auswertung von Kochrezepten mit Hilfe des Machine Learning-Algorithmus Word2Vec")      
        visualiser.empty()  
        visualiser = title.body()

    if tab == "Data Grabbing":
        visualiser.empty()

        def loadGrabImages():
            sample = "static/Rezept Beispiel.png"
            zubereitung = "static/Rezept Beispiel_Name.png"
            zutaten = "static/Rezept Beispiel_Zutaten.png"
            menge = "static/Rezept Beispiel_Menge.png"
            einheit ="static/Rezept Beispiel_Einheit.png"
            bezeichnung = "static/Rezept Beispiel_Bezeichnung.png"
            return [sample, zubereitung, zutaten, menge, einheit, bezeichnung]

        images = loadGrabImages()
        visualiser = grabbing.body(images[0],images[1], images[2], images[3], images[4], images[5])
    

    if tab == "Data Preprocessing":
        @st.cache
        def loadDataImages():
            emptyGrid = "static/Word2Vec Idee_leer.jpeg"
            milkGrid = "static/Word2Vec Idee_Milch.jpeg"
            cherryGrid = "static/Word2Vec Idee_Kirschen.jpeg"
            return [emptyGrid, milkGrid, cherryGrid]
        
        visualiser.empty()
        images = loadDataImages()
        visualiser = preprocessing.body(images[0], images[1], images[2])

    if tab == "Finaler Datensatz":
        visualiser.empty()
        visualiser = finaldata.body()
        
    if tab == "Modell Test":
        visualiser.empty()
        visualiser = w2vTest.body()
    
    if tab == "Modellauswertung":
        visualiser.empty()
        visualiser = auswertung.body()

    if tab == "Machine Learning Task":
        dataSelector = sidebar.dataSelector()
        text.empty()
        visualiser.empty()
        visualiser = mlTask.body(dataSelector)

    if tab == "Vectorization":
        visualiser.empty()
        visualiser = vectorization.body()

    if tab == "KMeans":
        visualiser.empty()
        kmeans = KMeans.KMeans()
        visualiser = kmeans.body()

    for i in range(13):st.sidebar.text("")
    st.sidebar.markdown("Interaktive Begleitdokumentation")
    st.sidebar.markdown("\"Data Mining: Extraktion von Onlinerezepten und Verarbeitung mittels Word2Vec\"")
    st.sidebar.text("Leon Kolyang Menkreo-Kuntzsch")
    st.sidebar.text("Microservices Repository:")
    st.sidebar.text("https://github.com/LeonKolyang/w2v-microservices")
    st.sidebar.text("Frontend Repository:")
    st.sidebar.text("https://github.com/LeonKolyang/w2v-Rezepte")

    

        
 
    

    

if __name__ == "__main__":
    main()