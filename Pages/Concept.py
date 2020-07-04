import streamlit as st

class Concept():
    def body(self, images):
        
        steps=["1. Zerlegung einer Zutat in seine Textelemente",
                    "2. Klassifizierung der Elemente mit Hilfe des Machine Learning Modells",
                    "3. Vereinheitlichung der Zutaten und Mengenangaben",
                    "4. Berechnung der Nährstoffe auf Grundlage der ermittelten Zutat und Menge"]
        select = st.selectbox("Schritt des Konzepts", steps)
        
        concept_steps = {step: image for step, image in zip(steps, images[:4])}
        st.image(concept_steps[select], use_column_width=True)

      