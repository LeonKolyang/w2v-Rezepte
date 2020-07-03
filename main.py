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
#import Doku_ingredientPrepare 
#import Doku_MLSequence
#import Preprocessing
#import MLTask
#import Grabbing
import Title
#import Concept
#import Vectorization
#import KMeans
from PIL import Image


import matplotlib.pyplot as plt


class Sidebar:
    def __init__(self):
        pass
        #self.DP = Doku_ingredientPrepare.DataProvider()

    def navigator(self):
        st.sidebar.header("DataHub")
        #routes = ["Projektvorstellung", "Konzept", "Data Grabbing", "Data Preprocessing", "Machine Learning Task", "Machine Learning Offline", "KMeans"]
        routes = ["Projektvorstellung", "Konzept", "Data Grabbing", "Data Preprocessing", "Vectorization", "KMeans"]

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




class APIHandler:
    def __init__(self):
        self.URL = "https://mlservice-dot-w2vrecipes.appspot.com/"
        self.testURL = "http://localhost:5000/"
    
    def startML(self, wordList, corpus, iterations):
        jsonWords = wordList.to_dict()
        jsonCorpus = corpus.to_dict()
        body = {    "wordList": jsonWords,
                    "corpus":   jsonCorpus,
                    "iterations": iterations}
        body = json.dumps(body)
        response = requests.post(self.testURL + "startml/", json = body)
        return response

    def getResults(self):
        try:
            response = requests.get(self.testURL + "progress/getresults")
            responseDict = response.json()
            return responseDict
        except:
            return None



def main():
    #DP = Doku_ingredientPrepare.DataProvider()
    title = Title.Title()
    #concept = Concept.Concept()
    #grabbing = Grabbing.Grabbing()
    #preprocessing = Preprocessing.Preprocessing()
    #mlTask = MLTask.MLTask()
    #vectorization = Vectorization.Vectorization()

    
    sidebar = Sidebar()
    visualiser = st.empty()
    saver = False

    try:
        MLdata = sidebar.opener()
    except:
        st.write("No Data available. Save Data first.")

    st.header("Data Grabbing und Preprocessing für Word2Vec")
    st.write("Vorbereitung einer Auswertung von Rezepten mit Hilfe des Machine Learning-Algorithmus Word2Vec")

    tab = sidebar.navigator()

    dataSelector = st.empty()
    data = None

    text = st.sidebar.empty()    

    if tab == "Projektvorstellung":
        visualiser.empty()

        @st.cache 
        def loadProjImages():
            projImg = "static/Idee.png"
            return projImg
        
        image = loadProjImages()
        visualiser = title.body(image)

    if tab == "Konzept":
        visualiser.empty()

        @st.cache
        def loadConImages():
            concept1 = "https://storage.googleapis.com/w2vfiles/static/Konzept_Zerlegung.png"
            concept2 = "https://storage.googleapis.com/w2vfiles/static/Konzept_Auswertung.png"
            concept3 = "https://storage.googleapis.com/w2vfiles/static/Konzept_Vereinheitlichung.png"
            concept4 = "https://storage.googleapis.com/w2vfiles/static/Konzept_Bewertung.png"
            return [concept1, concept2, concept3, concept4]
        
        images = loadConImages()
        
        visualiser = concept.body(images[0],images[1], images[2], images[3])

    if tab == "Data Grabbing":
        visualiser.empty()
        # sample = Image.open("Images/Rezept Beispiel.png")
        # zubereitung = Image.open("Images/Rezept Beispiel_Name.png")
        # zutaten = Image.open("Images/Rezept Beispiel_Zutaten.png")
        # menge = Image.open("Images/Rezept Beispiel_Menge.png")
        # einheit = Image.open("Images/Rezept Beispiel_Einheit.png")
        # bezeichnung = Image.open("Images/Rezept Beispiel_Bezeichnung.png")
        
        #@st.cache
        def loadGrabImages():
           
            sample = "https://storage.googleapis.com/w2vfiles/static/Rezept Beispiel.png"
            zubereitung = "https://storage.googleapis.com/w2vfiles/static/Rezept Beispiel_Name.png"
            zutaten = "https://storage.googleapis.com/w2vfiles/static/Rezept Beispiel_Zutaten.png"
            menge = "https://storage.googleapis.com/w2vfiles/static/Rezept Beispiel_Menge.png"
            einheit ="https://storage.googleapis.com/w2vfiles/static/Rezept Beispiel_Einheit.png"
            bezeichnung = "https://storage.googleapis.com/w2vfiles/static/Rezept Beispiel_Bezeichnung.png"
            return [sample, zubereitung, zutaten, menge, einheit, bezeichnung]

        images = loadGrabImages()
        visualiser = grabbing.body(images[0],images[1], images[2], images[3], images[4], images[5])
    

    if tab == "Data Preprocessing":
        #dataSelector = sidebar.dataSelector()
        #data = sidebar.dataReturn(dataSelector)
        emptyGrid = "https://storage.googleapis.com/w2vfiles/static/Word2Vec Idee_leer.jpeg"
        milkGrid = "https://storage.googleapis.com/w2vfiles/static/Word2Vec Idee_Milch.jpeg"
        cherryGrid = "https://storage.googleapis.com/w2vfiles/static/Word2Vec Idee_Kirschen.jpeg"
        
        visualiser.empty()
        #saver = sidebar.saveData()
        visualiser = preprocessing.body(emptyGrid, milkGrid, cherryGrid)
        
    

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

    

        
 
    

    

if __name__ == "__main__":
    main()