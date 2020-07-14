import streamlit as st
import pandas as pd
from Dataprocessing.Imageprovider import ImageLoader

class finalData():
    def __init__(self):
        self.images = ImageLoader.loadMLImages()

    def body(self):
        st.title("Finaler Datensatz")
        st.markdown("Das _Word2Vec_ _Modell_ wird auf dem erzeugten Zutatenverzeichnis und Korpus trainiert.")

        reducedWordList = pd.read_csv('Data/Doku_wordListNoAmount.csv', encoding="UTF-8", sep="|", header =None, names=["Wort"])
        reducedCorpus = pd.read_csv('Data/Doku_corpusNoAmount.csv', encoding="UTF-8", sep="|", header = None, names=["Zutat"])

        st.dataframe(reducedCorpus)
        st.text("Zutatenverzeichnis mit "+str(len(reducedCorpus))+" Einträgen")

        st.dataframe(reducedWordList)
        st.text("Korpus mit "+str(len(reducedWordList))+" Einträgen")