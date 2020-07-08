import streamlit as st
from Dataprocessing.Imageprovider import ImageLoader

class Title():
    def concept(self):
        images = ImageLoader.loadConImages()
        steps=["1. Zerlegung einer Zutat in seine Textelemente",
                    "2. Klassifizierung der Elemente mit Hilfe des Machine Learning Modells",
                    "3. Vereinheitlichung der Zutaten und Mengenangaben",
                    "4. Berechnung der Nährstoffe auf Grundlage der ermittelten Zutat und Menge"]
        select = st.selectbox("Schritt des Konzepts", steps)
        
        concept_steps = {step: image for step, image in zip(steps, images)}
        st.image(concept_steps[select], use_column_width=True)

    def mlConcept(self):
        grid = st.radio(label="Auswertung einer neuen Zutat durch ein trainiertes Word2Vec Modell",options=["Leeres Modell", "500 ml Rewe Beste Wahl Vollmilch", "1 Glas rote Kirschen"])
        images = ImageLoader.loadMLImages()
        if grid == "Leeres Modell": st.image(image=images[0], width=320, format = "JPEG")
        if grid == "500 ml Rewe Beste Wahl Vollmilch": st.image(image=images[1], width=320, format = "JPEG")
        if grid == "1 Glas rote Kirschen": st.image(image=images[2], width=320, format="JPEG")
            
        


    def body(self):
        images = ImageLoader.loadProjImages()
        for image in images:
            st.image(image, use_column_width=True)
        
        self.mlConcept()
