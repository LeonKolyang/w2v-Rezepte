import streamlit as st
class ImageLoader:
    @st.cache 
    def loadProjImages():
        projImg = ["static/Idee.png"]
        return projImg

    
    @st.cache
    def loadConImages():
        concept1 = "static/Konzept_Zerlegung.png"
        concept2 = "static/Konzept_Auswertung.png"
        concept3 = "static/Konzept_Vereinheitlichung.png"
        concept4 = "static/Konzept_Bewertung.png"
        return [concept1, concept2, concept3, concept4]

    @st.cache
    def loadMLImages():
        emptyGrid = "static/Word2Vec_Idee_leer.jpeg"
        milkGrid = "static/Word2Vec_Idee_Milch.jpeg"
        cherryGrid = "static/Word2Vec_Idee_Kirschen.jpeg"
        return [emptyGrid, milkGrid, cherryGrid]