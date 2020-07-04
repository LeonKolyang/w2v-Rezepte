import streamlit as st
from PIL import Image
import pandas as pd

class Grabbing:
    def __init__(self):
        pass        

    #@st.cache(allow_output_mutation=True)
    def body(self, sample, zubereitung, zutaten, menge, einheit, bezeichnung):
        st.title("Data Grabbing")
        
        st.header("Datenquelle _Rewe Rezeptwelt_")
        st.image(image=sample, use_column_width=True, caption="Beispiel eines Rezepts aus dem Rewe Rezeptekatalog")

        st.header("Aufbau der Datenbank ")
        st.write("In einem zweistufigen Verfahren wird aus den online angebotenen Rezepten eine Datenbank der Zutaten aufgebaut: "
                    "  \n"
                    "1. Abgreifen der einzelnen Rezepte und Entnahme der Zutaten"
                    "  \n"
                    "2. Überführung der Zutaten in eine eigene Datenbank"
                    "  \n")

        st.subheader("1. Abgreifen der einzelnen Rezepte und Entnahme der Zutaten")
        st.write("Über die HTML-Struktur können die einzelnen Elemente einer Webseite direkt ausgelesen werden. "
                    "So ist beispielsweise der Block für den Namen des Rezepts ein HTML-Element, die Liste mit den Zutaten ein weiteres, aber auch die Mengenangaben und Bezeichnungen der Zutaten können direkt ausgelesen werden. "
        )
        
        html = st.radio(label="HTML-Struktur", options=
                    ["<block: \"Name\">",
                    "<list:   \"Zutaten\">",
                    "<element 1-1: \"Menge\">",
                    "<element 1-2: \"Einheit\">",
                    "<element 1-3: \"Bezeichnung\">"]
                    )
        #htmlPic = st.empty()
        if html == ("<block: \"Name\">"):
            st.image(image=zubereitung, use_column_width=True)

        if html == "<list:   \"Zutaten\">":
            st.image(image=zutaten, use_column_width=True)

        if html ==  "<element 1-1: \"Menge\">":
            st.image(image=menge, use_column_width=True)

        if html ==  "<element 1-2: \"Einheit\">":
            st.image(image=einheit, use_column_width=True)

        if html ==  "<element 1-3: \"Bezeichnung\">":
            st.image(image=bezeichnung, use_column_width=True)

        example = {"ingredients":[{"name":"frische Spätzle (Kühlregal)","amount":"400","unit":"g"},{"name":"Salz","amount":"NoAmount","unit":"NoUnit"},{"name":"REWE ja! Emmentaler gerieben (45 % Fett)","amount":"100","unit":"g"},{"name":"Fett für die Form","amount":"NoAmount","unit":"NoUnit"},{"name":"REWE ja! Rohschinken gewürfelt","amount":"125","unit":"g"},{"name":"rote Zwiebel","amount":"1","unit":"NoUnit"},{"name":"REWE ja! Butter mild gesäuert","amount":"1","unit":"EL"},{"name":"Schnittlauch","amount":"5","unit":"Halm(e)"}]}
        st.write("Zu jedem Rezept hat sich so eine Struktur in folgender Form ergeben: ")
        st.write("Zutaten für **Käsespätzle mit Schinken**")

        sampleZutat = pd.DataFrame.from_dict(data=example["ingredients"])    
        sampleZutat = sampleZutat.rename(columns={"name": "Bezeichnung", "amount": "Menge", "unit": "Einheit"})
        st.write(sampleZutat)

        st.subheader("2. Überführung der Zutaten in eine eigene Datenbank")
        st.write("Die Zutaten aus den 5.245 abgegriffenen Rezepten werden in eine eigene Datenbank eingefügt. Daraus ergibt sich ein Bestand von 56.192 Zutaten, dieser umfasst allerdings Duplikate. Im folgenden Schritt des Preprocessing werden die Daten um Duplikate bereinigt, in die Bestandteile _Bezeichnung_, _Menge_ und _Einheit_ zerlegt und für den Machine Learning Algorithmus Word2Vec vorbereitet.")






        