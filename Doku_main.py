import streamlit as st
import pandas as pd
from pymongo import MongoClient
import csv
import time
import Doku_ingredientPrepare 
import Doku_MLSequence

class Sidebar:
    def __init__(self):
        self.DP = Doku_ingredientPrepare.DataProvider()

    def navigator(self):
        st.sidebar.header("Navigator")
        return st.sidebar.radio("Go to", ("Data Preprocessing", "Machine Learning Task"))
        

    def dataSelector(self):
        return st.sidebar.selectbox("Select Data Set",("Full Set", "Test Set"),index=1)

    def dataReturn(self, inputFile):
        if inputFile == "Full Set":
            return self.DP.dataCaller("Data/Doku_reweIngredients.csv")
        elif inputFile == "Test Set":
            return self.DP.dataCaller("Data/Doku_reweIngredientsSmall.csv")

    def reloadButton(self):
        return st.sidebar.button("Reload Data", key="r1")

    def saveData(self):
        return st.sidebar.button("Save Data from Preprocessing", key="dSave")

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



class TabHandler:
    def __init__(self):
        self.sidebar = Sidebar()
        self.DP = Doku_ingredientPrepare.DataProvider()
        self.recipeDfShort = None
        self.wordList = None
        self.corpus = None
        

    #@st.cache(allow_output_mutation=True)
    def preProcessing(self, inputFile):
        
        st.sidebar.text("")
        showDataframes = st.sidebar.checkbox("Show DataFrames", key="showFrames")

        st.write("Ziel: Mit Hilfe von Machine Learning automatisch aus einer Zutat eines Rezepts das konkrete Lebensmittel erkennen."
        "  \n"
        "Zutaten kommen beispielsweise in folgender Form vor: **500ml Rewe Milch von ja!**"
        "  \n"
        "Mit dem Lebensmittel: **Milch**"
        "  \n  \n"
        "Vorgehen:" 
        "  \n"
        "- Abgreifen der Rezepte von *https://www.rewe.de/rezepte/*""  \n"
        "Größe des Datensatzes: 5.245 Rezepte""  \n"
        "- Auslesen der Zutaten je Rezept und Trennen der Zutaten in *Menge*, *Einheit* und *Name*""  \n")


     
        recipeDf = inputFile


        if st.checkbox("Show DataFrame", key="initialDF", value=showDataframes):
            st.write(recipeDf)
        st.write("Größe des Datensatzes: ",len(recipeDf)," Zutaten (inkl. Duplikate)")
        
        st.subheader("Vorbereiten der Daten für eine Auswertung mit Word2Vec")
        st.write("In der finalen Anwendung eines ML-Algorithmus (Word2Vec), soll der Algorithmus die Zuordnung Zutat -> Lebensmittel erkennen und automatisch vornehmen. "
        "Dafür werden zunächst einzelne Lebensmittel Bezeichnungen aus dem Datensatz ermittelt und anschließend den Zutaten zugeordnet. "
        "So soll ein erster Trainings- und Testdatensatz geschaffen werden, welcher dem ML-Algorithmus mit den zugeprdneten Lebensmitteln eine Kontrolllösung bietet.")
        
        recipeDf = self.DP.duplicateReduce(recipeDf)

        st.write("- Bereinigung der Zutaten um Duplikate"
        "  \n"
        "Größe des Datensatzes: ",len(recipeDf)," Zutaten")
        if st.checkbox("Show DataFrame", key=0, value=showDataframes):
            st.write(recipeDf)

        ingredientDict = self.DP.topIngredients()

        st.write("  \n"
        "- Entnehmen der Top ",len(ingredientDict)," Lebensmittel")
        if st.checkbox("Show DataFrame", key=1, value=showDataframes):
            st.write(ingredientDict)
        self.DP.ingredientList = ingredientDict.to_list()

        st.write("- Zuordnen der top ", len(ingredientDict), " Lebensmittel zu den vorhandenen Rezepten")
        recipeDf = self.DP.ingredientMatch(recipeDf)
        if st.checkbox("Show DataFrame", key=2, value=showDataframes):
            st.write(recipeDf)

        self.recipeDfShort = self.DP.listCutter(recipeDf)

        st.write("- Filtern von Zutaten ohne/ mit uneindeutiger Lebensmittelbezeichnung, also Menge an Zuordnungen ungleich 1"
        "  \n"
        "Größe des Datensatzes: ", len(self.recipeDfShort), " Zutaten")
        if st.checkbox("Show DataFrame", key=3, value=showDataframes):
            st.write(self.recipeDfShort) 
        # if st.checkbox("Save DataFrame"):
        #     recipeDfShort.to_csv("Data/Doku_ZutatenLebensmittel.csv",index=False)
        #     st.write("Zuordnung gespeichert")

        self.recipeDfShort = self.recipeDfShort[["name", "unit"]]
        st.write("- Erstellen einer Liste mit allen Wörtern, fehlende Werte (_nan_) ausgenommen")

        self.wordList = self.DP.createWordList(self.recipeDfShort)
        
            
        if st.checkbox("Show DataFrame", key=4, value=showDataframes):
            st.dataframe(self.wordList)
        
        st.write("- Corpus der Rezepte")
        self.corpus = self.DP.createCorpus(self.recipeDfShort)
        if st.checkbox("Show DataFrame", key=5, value=showDataframes):
            st.dataframe(self.corpus)


  

    def machineLearning(self, inputFile, MLdata):
        vectors = None
        wordList = MLdata["wordList"]
        corpus = MLdata["corpus"]
        st.subheader("Klassifizierung der Bestandteile einer Zutat mit Word2Vec")
        st.write(   "Die Daten wurden in den vorherigen Schritten vorbereitet, um mit einem Machine Learning Algorithmus ausgewertet zu werden."
                    "Im weiteren Vorgehen wird mit TensorFlow der Word2Vec Algorithmus umgesetzt und versucht, die Beziehungen zwischen den einzelnen Zutaten abzubilden"
                    "und diese in möglichen weiteren Schritten final auszuwerten.")
        mlP = Doku_MLSequence.MLParser(wordList, 8)

        sentences = self.DP.extractSentences(corpus)
        neighborDf = mlP.calculateNeighbors(sentences)

        word2int = mlP.createWordVector()
        mlP.declareLoss(neighborDf, word2int)
        iterations = st.number_input("Anzahl Iterationen", value = 20000)

        if st.button("Start Session", key="train"):
            mlP.train()
            
            sess = mlP.startSession(iterations)

            vectors = mlP.calculateVectors(sess)

        if vectors is not None:
            vectorDf = pd.DataFrame(vectors)
            vectorDf.to_csv("Data/Doku_vectors.csv", index=False)
            w2v = mlP.vectorToDf(vectors, wordList)
            plot = mlP.plot(w2v, vectors)
            st.pyplot(plot)

        if st.button("Show results from previous run", key="prev"):
            vectorDf = pd.read_csv("Data/Doku_vectors.csv")
            wordList = pd.read_csv('Data/Doku_wordList.csv', sep ="§", encoding = "UTF-8", header = None)
            loss = None
            with open("Data/loss.txt", "r") as lossfile:
                loss = lossfile.readline()
            st.markdown("Loss: "+loss)
            vectorDf.columns = ["x1", "x2"]
            w2v = mlP.reloadVectorToDf(vectorDf, wordList)
            plot = mlP.plot(w2v, vectorDf)
            st.pyplot(plot)







def main():
    DP = Doku_ingredientPrepare.DataProvider()
    sidebar = Sidebar()
    tabHandler = TabHandler()
    visualiser = st.empty()
    saver = False
    try:
        MLdata = sidebar.opener()
    except:
        st.write("No Data available. Save Data first.")

    st.title("Rewe Rezepte")
    st.subheader("Auswertung von Rezepten mit Hilfe des Machine Learning-Algorithmus Word2Vec")

    tab = sidebar.navigator()

    dataSelector = sidebar.dataSelector()
    data = sidebar.dataReturn(dataSelector)
    
    #reload = sidebar.reloadButton()

    text = st.sidebar.empty()
    
    
    #if reload:
    #    data = None
    #    data = DP.importer(dataSelector)

    if tab == "Data Preprocessing":
        visualiser.empty()
        saver = sidebar.saveData()
        visualiser = tabHandler.preProcessing(data)

    if tab == "Machine Learning Task":
        text.empty()
        visualiser.empty()
        opener = sidebar.openData()
        if opener:
            text.success("WordList and Corpus loaded!")
            time.sleep(2)
            text.empty()
            MLdata = sidebar.opener()
        visualiser = tabHandler.machineLearning(data,MLdata)
    
    if saver:
        sidebar.saver({ "wordList": tabHandler.wordList,
                        "corpus":   tabHandler.corpus})
        text.success("WordList and Corpus saved!")
        time.sleep(2)
        text.empty()
        
        
 
    

    

if __name__ == "__main__":
    main()