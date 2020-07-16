import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Auswertung():
    def plotting(self, df, cluster, focus=False):
        fig, ax = plt.subplots()

        if not(focus):
            x =  df["x"][df["Cluster"].isin(cluster)]
            y =  df["y"][df["Cluster"].isin(cluster)]

            ax.scatter(x, y)
    
        def annotate(row):
            if focus:
                ax.annotate(row["labels"], (row["x"], row["y"]))
            else:
                if row["Cluster"] not in cluster:
                    ax.annotate(".", (row["x"], row["y"]))
        df.apply(lambda row: annotate(row), axis=1)

        PADDING = 1.0
        x_axis_min = np.amin(df, axis=0)[0] - PADDING
        y_axis_min = np.amin(df, axis=0)[1] - PADDING
        x_axis_max = np.amax(df, axis=0)[0] + PADDING
        y_axis_max = np.amax(df, axis=0)[1] + PADDING
        
        plt.xlim(x_axis_min,x_axis_max)
        plt.ylim(y_axis_min,y_axis_max)
        plt.rcParams["figure.figsize"] = (10,10)

        
        return plt


    def body(self):

        st.title("Auswertung")
        st.markdown("Die ausgewerteten Wörter werden über ein Clustering in Gruppen eingeteilt. "
                    "Ziel des Clusterings ist es, Gruppen zu identifizieren, welche ausschließlich _Bezeichnungen_ oder keine _Bezeichnungen_ enthalten. "
                    "Der Anteil an _Bezeichnungen_ in einem Cluster wird über die _Reinheit_ angegeben. "
                    "Je größer die _Reinheit_, desto mehr _Bezeichnungen_ sind enthalten.")

        bezeichnung_list = pd.read_csv("Data/topIngredients.csv", header = 0,sep="," )["name"].tolist()

        no_iterations = st.sidebar.slider("Anzahl Trainingsepochen", min_value=1, value= 5, max_value=10)
        window_size = st.sidebar.slider("Wortfenstergröße", min_value=1, value=2, max_value=10)
        dimensions = st.sidebar.slider("Dimensionen", min_value=1, value=300, max_value=1200,step=300)
        no_cluster = st.sidebar.selectbox("Anzahl Cluster",options=[2,5,10,20,40,50,70,100,200], index=2 )

                                
        try:
            w2v = pd.read_csv("Data/w2v_full_results_"+str(no_iterations)+"_"+str(window_size)+"_"+str(no_cluster)+"_"+str(dimensions)+".csv", header=0, sep="|", index_col=0)

            cluster_results= pd.read_csv("Data/w2v_cluster_results_"+str(no_iterations)+"_"+str(window_size)+"_"+str(no_cluster)+"_"+str(dimensions)+".csv", header=0, sep="|", index_col=0)        
        except OSError as err:
            #st.write("OS error: {0}".format(err))
            st.info("Modelltest mit den gewählten Parametern noch nicht durchgeführt.")
            return

        w2v["Cluster"] = w2v["Cluster"].apply(lambda x: "Cluster "+str(x))

        st.dataframe(cluster_results.sort_values(["Reinheit"], ascending=False))
        clusterlist = list(w2v["Cluster"].sort_values().unique())
        default_cluster = cluster_results.loc[cluster_results["Reinheit"]>70]
        default_cluster = list(default_cluster.index)
        
        st.write("")
        st.markdown("Über die folgende Selektion lassen sich Details zu einem oder mehreren Clustern anzeigen.")

        cluster = st.multiselect("Details zu", clusterlist, default_cluster)
        #cluster = st.selectbox("Details zu", clusterlist)
        w2v_filtered = w2v[w2v["Cluster"].isin(cluster)]


        w2v_styled = w2v_filtered.style.apply(lambda x: ["background: lightgreen" if (set(bezeichnung_list).intersection(x.values)) else "" for i in x], axis = 1)

        #st.dataframe(w2v.loc[w2v["Cluster"] == cluster])
        st.dataframe(cluster_results[cluster_results.index.isin(cluster)][["Zugeordnete Wörter", "Daraus Bezeichnungen","Reinheit"]])
        st.dataframe(w2v_styled)

        st.write("Lage des Clusters in der gesamten Datenmenge:")
        plt.clf()
        plot_big = self.plotting(w2v, cluster)
        st.pyplot(plot_big)

        focus_cluster = st.selectbox("Fokus auf", clusterlist)
        w2v_focused = w2v.loc[w2v["Cluster"] == focus_cluster]
        plt.clf()
        plot_focused = self.plotting(w2v_focused, focus_cluster, True)
        st.pyplot(plot_focused)




      