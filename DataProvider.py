import streamlit as st
import csv
import numpy as np
import pandas as pd
import time
from pymongo import MongoClient

class DataProvider:
    def __init__(self):
        self.recipeDf = None
        self.ingredientList = None

        
    def importer(self, inputFile):
        client = MongoClient('localhost:27017')
        db = client.Recipes["reweIngredientsStripped"]
        recipeDf = pd.DataFrame(columns = ["amount", "unit","name"])
        ind = 0

        bar = st.progress(0)
        limit = 0
        recipes = list(db.find())

        if inputFile == "Test Datensatz":
            output = "Data/Doku_reweIngredientsSmall.csv"
            recipes = recipes[:100]
        elif inputFile == "Vollständiger Datensatz":
            output = "Data/Doku_reweIngredients.csv"
        else:
            st.write("Data Select Error  \n", inputFile)

        for recipe in recipes:
            del recipe["_id"]
            del recipe["nameComponents"]
            if recipe["unit"] == "NoUnit": 
                recipe["unit"] = ""
            if recipe["amount"] == "NoAmount": 
                recipe["amount"] = ""
            for key, element in recipe.items():
                recipe[key] = [element]
                df = pd.DataFrame(data=recipe)
            recipeDf = recipeDf.append(df, ignore_index=True)
            ind += 1
            prog = (ind/len(recipes))
            if prog <= 1: 
                bar.progress(prog)
                
        recipeDf.to_csv(output, index = False)
        bar.empty()

        return recipeDf

    @st.cache(persist=True)
    def checkZutaten(self, inputFile):
        try:
            recipeDf = pd.read_csv(inputFile, encoding="UTF-8")
            return recipeDf
            

        except:
            recipeDf = importer(inputFile)
            return recipeDf


    def dataCaller(self, inputFile):
        recipeDf = pd.DataFrame(columns = [ "Menge", "Einheit","Name"])
        recipeDf = self.checkZutaten(inputFile)
        return recipeDf


    def duplicateReduce(self, recipeDf):
        recipeDf = recipeDf.sort_values("name")
        recipeDf = recipeDf.drop_duplicates(subset = "name", keep=False)
        return recipeDf

    def topIngredients(self):
        ingredientDict = pd.read_csv("Data/topIngredients.csv", encoding="UTF-8")["name"]
        return ingredientDict

    
    def matchHelper(self, ingredient):
        list = ingredient["name"].split(" ")
        ingList = []
        for word in list:
            if word in self.ingredientList:
                ingList.append(word)
        return ingList

    def ingredientMatch(self, recipeDf):
        recipeDf["ingredient"] = recipeDf.apply(self.matchHelper, axis=1)
        return recipeDf
    
    def cutLists(self, x):
        return x[0]

    @st.cache(persist=True)
    def listCutter(self, recipeDf):
        recipeDfShort = recipeDf[recipeDf["ingredient"].str.len()==1 ]
        recipeDfShort.loc[:,("ingredient")] = recipeDfShort["ingredient"].apply(self.cutLists)
        return recipeDfShort
    
    @st.cache(persist=True)
    def createWordList(self, recipeDf):
        wordList = []
        for index, row in recipeDf.iterrows():
            columnList = [row["amount"], row["unit"],  row["name"]]
            #columnList = [row["unit"],  row["name"]]
            for column in columnList:
                if type(column) == str:
                    list = column.split()
                    for word in list:
                        if word not in wordList:
                            wordList.append(word)
        return wordList
    
    @st.cache(persist=True)
    def createCorpus(self, recipeDf):
        corpus = []
        for index, row in recipeDf.iterrows():
            columnList = [row["amount"], row["unit"],  row["name"]]
            #columnList = [row["unit"],  row["name"]]
            corp = ""
            for column in columnList:
                if type(column) == str:
                    corp += column + " "
            corpus.append(corp)
        return corpus

    def extractSentences(self, corpus):
        sentences = []
        for index, row in corpus.iterrows():
            columnList = [row[0]]
            for sentence in columnList:
                #st.write(sentence)
                sentences.append(sentence.split())
        return sentences
    
    def clean_wordlist_signs(self, wordList, signList):
        clean_check = False

        def stripper(x, signList):
            sep = ""
            stripped_x = x.strip(sep.join(signList))
            if x == stripped_x: clean_check=True
            return stripped_x

        wordList_clean = wordList.apply(lambda x: stripper(x[0], signList), axis=1)
        return wordList_clean 

