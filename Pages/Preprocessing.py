import streamlit as st
import pandas as pd
from PIL import Image

from Dataprocessing import DataProvider

class Preprocessing:
    def __init__(self):
        self.DP = DataProvider.DataProvider()
        
        

    #@st.cache(allow_output_mutation=True)
    def body(self, emptyGrid, milkGrid, cherryGrid):
        
        st.sidebar.text("  \n")
        st.sidebar.text("  \n")
        showDataframes = st.sidebar.checkbox("Alle Datensätze anzeigen", key="showFrames")

        st.title("Data Preprocessing")
        st.header("Aufbereiten der Rezeptdaten für Word2Vec")     
        
        st.subheader("Duplikate entfernen")
        st.write("Primäre Anforderung an die Daten für den Word2Vec Algorithmus ist ein Korpus und eine Wortliste aus den Zutatendaten. "
                    "Um diese zu erstellen wird zunächst der Datensatz um Duplikate bereinigt. "
                    "Aus den 56.192 Zutaten bleiben nach der Bereinigung noch 6.279 einzigartige Zutaten. " )

        recipeDf = pd.read_csv("Data/Doku_reweIngredients.csv", encoding="UTF-8")

        zutatenDf = recipeDf
        zutatenDf = zutatenDf.rename(columns={"name": "Bezeichnung", "amount": "Menge", "unit": "Einheit"})
        st.write("Größe des originalen Datensatzes: ",len(zutatenDf)," Zutaten (inkl. Duplikate)")
        if st.checkbox("Datensatz anzeigen", key="initialDF", value=showDataframes):
            st.write(zutatenDf[["Menge", "Einheit", "Bezeichnung"]])
            st.text("Originaler Datensatz")

        with st.echo():
            zutatenDf = zutatenDf.sort_values("Bezeichnung")
            zutatenDf = zutatenDf.drop_duplicates(subset = "Bezeichnung", keep=False)

        recipeDf = self.DP.duplicateReduce(recipeDf)

        zutatenDf = recipeDf
        zutatenDf = zutatenDf.rename(columns={"name": "Bezeichnung", "amount": "Menge", "unit": "Einheit"})
        zutatenDf.sort_values("Bezeichnung")
        st.write("Größe des Datensatzes nach Bereinigung um Duplikate: ",len(zutatenDf)," Zutaten")
        if st.checkbox("Datensatz anzeigen", key=0, value=showDataframes):
            st.write(zutatenDf[["Menge", "Einheit", "Bezeichnung"]])
            st.text("Bereinigter Datensatz")

        st.subheader("Zuordnung der Bezeichnung")
        st.write("Um den Datensatz für eine spätere automatisierte Überprüfung des Machine Learning Algorithmus zu qualifizieren, werden den Zutaten händisch die entsprechenden Bezeichnungen zugeordnet. Für eine erste Auswertung werden die einzelnen Zutaten nach ihrer Häufigkeit sortiert und für die häufigsten 500 Zutaten die _Bezeichnungen_ in eine separate Liste eingetragen.")

        with st.echo():
            topZutaten = zutatenDf["Bezeichnung"]
            topZutaten = topZutaten.sort_values()
        
        ingredientDict = self.DP.topIngredients()
        
        topZutaten = ingredientDict
        topZutaten = topZutaten.rename("Bezeichnung")
        #topZutaten.to_csv("../work_Data/top_zutaten.csv", sep="|", index = False)

        if st.checkbox("Datensatz anzeigen", key=1, value=showDataframes):
            st.write(topZutaten)
            st.text("Liste mit den Top " + str(len(topZutaten)) +" Bezeichnungen")

        st.write("Diese Liste beinhaltet die gefilterten Bezeichnungen und kann verwendet werden, um den restlichen Zutaten die Bezeichnung zu entnehmen. Für die restlichen Zutaten wird dabei abgeglichen, welches Bezeichnung aus der Top 500 Liste sich in der jeweiligen Zutat wiederfindet. Um die folgenden Schritte zu vereinfachen und ein deutlicheres Ergebnis zu erzielen, werden dabei Zutaten mit mehrfach zuordenbarer Bezeichnung ausgefiltert. \"Vollmilch Schokolade\" beispielsweise entfällt, da hier sowohl „Vollmilch“ als auch „Schokolade“ als Bezeichnung erkannt werden."
                    )

      
        self.DP.ingredientList = ingredientDict.to_list()

        recipeDf = self.DP.ingredientMatch(recipeDf)

        recipeDfShort = self.DP.listCutter(recipeDf)

        with st.echo():
            def matchHelper(ingredient):
                list = ingredient["Bezeichnung"].split(" ")
                ingList = []
                for word in list:
                    if word in topZutaten:
                        ingList.append(word)
                return ingList
            
            zutatenDf["Bezeichnung"] = zutatenDf.apply(matchHelper, axis=1)

        zutatenDf = recipeDfShort
        zutatenDf = zutatenDf.rename(columns={"name": "Bezeichnung", "amount": "Menge", "unit": "Einheit","ingredient":"Zuordnung"})
        #zutatenDf.to_csv("../work_Data/zutatenDf.csv", sep="|", index=False)
        st.write(" Das Ergebnis ist ein Datensatz mit ", len(zutatenDf), " Zutaten.")
        if st.checkbox("Datensatz anzeigen", key=3, value=showDataframes):
            st.write(zutatenDf[["Menge", "Einheit", "Bezeichnung", "Zuordnung"]]) 

        st.subheader("Generieren des Korpus und der Wortliste")
        st.write("Aus dem bereinigten Datensatz kann anschließend der Korpus und die Wortliste entnommen werden. "
                    "Für den Korpus werden Zeilen des Datensatzes zusammengefasst und in eine Liste übergeben. "
        )

        fullWordList = pd.read_csv('Data/Doku_wordListAllColumnns.csv', encoding="UTF-8", sep="|", header =None, names=["Wort"])
        fullCorpus = pd.read_csv('Data/Doku_corpusAllColumns.csv', encoding="UTF-8", sep="|", header = None, names=["Zutat"])
        
        reducedWordList = pd.read_csv('Data/Doku_wordListNoAmount.csv', encoding="UTF-8", sep="|", header =None, names=["Wort"])
        reducedCorpus = pd.read_csv('Data/Doku_corpusNoAmount.csv', encoding="UTF-8", sep="|", header = None, names=["Zutat"])

        if st.checkbox("Korpus anzeigen",value=showDataframes): 
            st.dataframe(reducedCorpus)
            st.text("Reduzierter Korpus mit "+str(len(reducedCorpus))+" Einträgen")

       

        if st.checkbox("Wortliste anzeigen",value=showDataframes):
            st.dataframe(reducedWordList)
            st.text("Reduzierte Wortliste mit "+str(len(reducedWordList))+" Einträgen")

        st.write("Bevor der Korpus und die Wortliste durch den Algorithmus ausgewerter werden, wird die Wortliste um Sonderzeichen und Duplikate bereinigt.")
        st.write("Es werden nur Sonderzeichen an erster oder letzter Stelle des Wortes berücksichtigt.")

        signList = [",", "(", ")", "/", ".", "+","\"", ":","-","„","“","&"]
        reducedCleanWordList = self.DP.clean_wordlist_signs(reducedWordList, signList)

        if st.checkbox("Datensatz anzeigen", key=4, value=showDataframes):
            st.dataframe(reducedCleanWordList)
            st.text("Reduzierter Korpus mit "+str(len(reducedCleanWordList))+" Einträgen")

        st.write("Anschließend werden noch Duplikate aus der Wortliste entfernt.")

        reducedCleanWordList = reducedCleanWordList.sort_values()
        reducedCleanWordList = reducedCleanWordList.drop_duplicates(keep=False)

        if st.checkbox("Datensatz anzeigen", key=5, value=showDataframes):
            st.write(reducedCleanWordList)
            st.text("Reduzierter Korpus mit "+str(len(reducedCleanWordList))+" Einträgen")        

        st.write("Der Korpus und die Wortliste werden im nächsten Schritt an den Algorithmus übergeben und dienen als Grundlage zur Ausführung.")

        

        

        # st.write(recipeDfShort)
        # wordListFull = self.DP.createWordList(recipeDfShort)
        # wordSeries = pd.Series(wordListFull)
        # wordSeries.to_csv('Data/Doku_wordListAllColumnns.csv',header=False, index=False,sep="|")
        # st.write(wordSeries)

        # corpus = self.DP.createCorpus(recipeDfShort)
        # corpusSeries = pd.Series(corpus)
        # corpusSeries.to_csv('Data/Doku_corpusAllColumns.csv',header=False, index=False,sep="|")
        # st.write(corpusSeries)

      

        
        # if st.checkbox("Save DataFrame"):
        #     recipeDfShort.to_csv("Data/Doku_ZutatenLebensmittel.csv",index=False)
        #     st.write("Zuordnung gespeichert")

        # recipeDfShort = recipeDfShort[["name", "unit"]]
        # st.write("- Erstellen einer Liste mit allen Wörtern, fehlende Werte (_nan_) ausgenommen")

        # wordList = self.DP.createWordList(recipeDfShort)
        
        # wordSeries = pd.Series(wordList)
        # wordSeries.to_csv('Data/Doku_wordListNoAmount.csv',header=False, index=False,sep="|")
        
        # if st.checkbox("Show DataFrame", key=4, value=showDataframes):
        #     st.dataframe(wordList)
        
        # st.write("- Corpus der Rezepte")
        # corpus = self.DP.createCorpus(recipeDfShort)

        # corpusSeries = pd.Series(corpus)
        # corpusSeries.to_csv('Data/Doku_corpusNoAmount.csv',header=False, index=False,sep="|")
        
        # if st.checkbox("Show DataFrame", key=5, value=showDataframes):
        #     st.dataframe(corpus)


  


