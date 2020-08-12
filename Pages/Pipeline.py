import pandas as pd
import streamlit as st
import numpy as np
import requests 
import json
from Dataprocessing.Imageprovider import ImageLoader


class Pipeline():

    def body(self):
        
        new_phrase = st.text_input("Zutat")

        if st.button("Neue Zutat auswerten"):
            data = {"data":{"new_phrase": new_phrase.split()}}
            data = json.dumps(data)
            new = requests.get( "https://w2v-mlmodels.herokuapp.com/evaluate_new_phrase/word2vec", data=data)
            new = json.loads(new.json())
            new = pd.DataFrame(new)
            new.to_csv("new_phrase.csv", index=None, sep="|")
            new = new.rename(columns={"reinheit":"Bezeichnung", "labels":"Wort"})
            st.write("Als Bezeichnung identifiziert:")
            st.write(new.loc[new["Bezeichnung"]==new["Bezeichnung"].max()]["Wort"])
            st.write("Sonstige Zuordnungen:")
            st.write(new[["Wort", "Bezeichnung"]])

        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("Das W2V-Rezepte Tool wertet Phrasen aus und erkennt das enthaltene Lebensmittel. "
                "Die Auswertung basiert auf folgendem Schema und wird unter dem Tab \"Projektvorstellung\" genauer erklärt.")
        self.mlConcept()

    def mlConcept(self):
        grid = st.radio(label="Auswertung einer neuen Zutat durch ein Word2Vec Modell",options=["Trainiertes Modell", "500 ml Rewe Beste Wahl Vollmilch", "1 Glas rote Kirschen"])
        i1, i2, i3 = ImageLoader.loadMLImages()
        if grid == "Trainiertes Modell": st.image(image=i1, width=320, format = "JPEG")
        if grid == "500 ml Rewe Beste Wahl Vollmilch": st.image(image=i2, width=320, format = "JPEG")
        if grid == "1 Glas rote Kirschen": st.image(image=i3, width=320, format="JPEG")
            
    def start_service(self):
            zutatenverzeichnis = pd.read_csv("BaseData/zutatenverzeichnis.csv", header=None,sep="|")
            zutatenverzeichnis.columns=["name"] 
            top_ingredients = pd.read_csv("BaseData/top_ingredients.csv", header=0, index_col=0)
            top_ingredients = top_ingredients.reset_index(drop=True)

            parameters = {"iterations": 10,
                            "window_size" : 2,
                            "dimensions" : 300,
                            "min" : 0,
                            "neg" : 0}
            # parameters = {"iterations": st.number_input("iterations", value=10),
            #                 "window_size" : st.number_input("window_size", value=2),
            #                 "dimensions" : st.number_input("dimensions", value=300),
            #                 "min" : st.number_input("min", value=5),
            #                 "neg" : st.number_input("neg", value=5)}

            parameter_list = []
            for parameter, value in parameters.items():
                parameter_list.append({"parameter": parameter, "value": value} )

            parameters={"no_clusters": 8}
            #parameters={"no_clusters":st.number_input("Cluster", value=8)}

            clparameter_list = []
            for parameter, value in parameters.items():
                clparameter_list.append({"parameter": parameter, "value": value} )     

            url = "https://w2v-mlmodels.herokuapp.com/create_model/word2vec/new"
            put = requests.post(url)

            zutatenverzeichnis = zutatenverzeichnis.to_dict()
            data_dict = {"data": {"zutatenverzeichnis": zutatenverzeichnis}}
            data = json.dumps(data_dict)
            requests.put( "https://w2v-mlmodels.herokuapp.com/load_model_data/word2vec", data=data)

            data = parameter_list
            data = json.dumps(data)
            requests.put("https://w2v-mlmodels.herokuapp.com/load_model_parameters/word2vec", data=data)

            mlurl = "https://w2v-mlmodels.herokuapp.com/run_model/word2vec"
            put = requests.put(mlurl)
            
            url = "https://w2v-mlmodels.herokuapp.com/get_result/word2vec/all"
            get = requests.get(url)
            word_vectors = json.loads(get.json())
            word_vectors = pd.DataFrame(word_vectors)

            requests.post("https://w2v-mlmodels.herokuapp.com/create_model/clustering/new")
            word_vectors = word_vectors.to_dict()
            top_ingredients = top_ingredients.to_dict()
            data_dict = {"data": {"word_vectors": word_vectors,
                                    "ingredients": top_ingredients}}
            data = json.dumps(data_dict)
            requests.put( "https://w2v-mlmodels.herokuapp.com/load_model_data/clustering", data=data)

            mlurl = "https://w2v-mlmodels.herokuapp.com/load_model_parameters/clustering"
            params = clparameter_list
            params = json.dumps(params)
            put = requests.put(mlurl, data=params)

            mlurl = "https://w2v-mlmodels.herokuapp.com/run_model/clustering"
            put = requests.put(mlurl)