import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Auswertung():
    def plotting(self, df, cluster):
        fig, ax = plt.subplots()

        def annotate(row):
            if row["Cluster"]==cluster:
                ax.annotate(row["labels"], (row["x"], row["y"]))
            else:
                ax.annotate(".", (row["x"], row["y"]))


        df.apply(lambda row: annotate(row), axis=1)
        # for word, x1, x2 in zip(df['labels'], df['x'], df['y']):
        #     ax.annotate(word, (x1,x2 ))

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
        bezeichnung_list = pd.read_csv("Data/topIngredients.csv", header = 0,sep="," )["name"].tolist()

        no_iterations = st.sidebar.number_input("Anzahl Trainingsepochen", min_value=1, value= 5)
        window_size = st.sidebar.number_input("Wortfenstergröße", min_value=1, value=2)
        no_cluster = st.sidebar.number_input("Anzahl Cluster", min_value=1, value= 5)

                                
        try:
            w2v = pd.read_csv("Data/w2v_full_results_"+str(no_iterations)+"_"+str(window_size)+"_"+str(no_cluster)+".csv", header=0, sep="|", index_col=0)

            cluster_results= pd.read_csv("Data/w2v_cluster_results_"+str(no_iterations)+"_"+str(window_size)+"_"+str(no_cluster)+".csv", header=0, sep="|", index_col=0)        
        except OSError as err:
            #st.write("OS error: {0}".format(err))
            st.write("Modelltest mit den gewählten Parametern noch nicht durchgeführt.")
            return

        w2v["Cluster"] = w2v["Cluster"].apply(lambda x: "Cluster "+str(x[1]))

        st.dataframe(cluster_results)
        clusterlist = w2v["Cluster"].sort_values().unique()

        cluster = st.selectbox("Details zu", clusterlist)

        w2v_filtered = w2v.loc[w2v["Cluster"] == cluster]

        w2v_styled = w2v_filtered.style.apply(lambda x: ["background: lightgreen" if (set(bezeichnung_list).intersection(x.values)) else "" for i in x], axis = 1)

        #st.dataframe(w2v.loc[w2v["Cluster"] == cluster])
        st.dataframe(cluster_results.loc[cluster_results.index == cluster][["Zugeordnete Wörter", "Daraus Bezeichnungen","Reinheit"]])
        st.dataframe(w2v_styled)

        
        plt.clf()
        plot = self.plotting(w2v, cluster)
        st.pyplot(plot)




      