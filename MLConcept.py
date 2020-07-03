import streamlit as st

class MLConcept():
    def body(self, images):
        grid = st.radio(label="Auswertung einer neuen Zutat durch ein trainiertes Word2Vec Modell",options=["Leeres Modell", "500 ml Rewe Beste Wahl Vollmilch", "1 Glas rote Kirschen"])
        if grid == "Leeres Modell": st.image(image=images[0], width=320, format = "JPEG")
        if grid == "500 ml Rewe Beste Wahl Vollmilch": st.image(image=images[1], width=320, format = "JPEG")
        if grid == "1 Glas rote Kirschen": st.image(image=images[2], width=320, format="JPEG")
        
     