#import streamlit as st
import pandas as pd
import math
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from statistics import mean
from random import randint
import numpy as np
import copy
from sklearn.cluster import KMeans as KMeansSK

#Klasse zur Ausführung des KMeans Algorithmus
class KMeans():
    def __init__(self):
        pass

    #Generiere die Cluster auf den ersten n Punkten
    def introduce_cluster(self, no_of_clusters, df):
        cluster_dict = {str(i): (df.loc[i]["x"], df.loc[i]["y"]) for i in range(no_of_clusters)}
        return cluster_dict

    #Berechne die Distanz zwischen einem Punkt und den Clustern
    def calculate_distances(self, clusters, df):
        def distance(x,y, cluster):
            euclidian_distance = math.sqrt((x-cluster[0])**2+(y-cluster[1])**2)
            return euclidian_distance

        for cluster, position in clusters.items():
            df[str(cluster)] = df.apply(lambda row: distance(row[0], row[1], position), axis=1)
        
        return df

    #Ordne die Cluster den Punkten zu
    def assign_clusters(self, cluster_dict, df):
        clusterlist = list(cluster_dict.keys())

        def assign(values, clusterlist):
            valuelist =  [values[value] for value in clusterlist]
            #Zuordnung nach kürzester Distanz
            assignedCluster = clusterlist[valuelist.index(min(valuelist))]
            return assignedCluster
            
        df["assigned to"] = df.apply(lambda row: assign(row, clusterlist) , axis = 1)
        return df

    #Zentriere die Cluster an dem Mittelpunkt ihrer Punkte
    def centroid_coordinates(self, cluster_dict, df):
        clusterlist = list(cluster_dict.keys())

        x_mean = [mean(df.loc[df["assigned to"] == cluster]["x"]) for cluster in clusterlist]
        y_mean = [mean(df.loc[df["assigned to"] == cluster]["y"]) for cluster in clusterlist]


        for i, cluster in enumerate(clusterlist):

            cluster_dict[cluster] = (x_mean[i], y_mean[i])

        return cluster_dict

    #Verfolge die Änderung der Cluster, um eine Abbruchfunktionumzusetzen
    def check_cluster_change(self, cluster_dict_before , cluster_dict_after, threshold = 1):
        
        change = [(abs(coordinates[0] - cluster_dict_after[cluster][0]) +
                                    abs(coordinates[1] - cluster_dict_after[cluster][1]) )
                                    for cluster, coordinates in cluster_dict_before.items()]
        threshold_hit = [True if (  abs(coordinates[0] - cluster_dict_after[cluster][0]) +
                                    abs(coordinates[1] - cluster_dict_after[cluster][1]) ) <= threshold
                                    else False
                                    for cluster, coordinates in cluster_dict_before.items() ]
        
        #Ist die Änderung aller Cluster kleiner als der Threshold, brich den Lauf ab
        if sum(threshold_hit) == len(cluster_dict_before):
            return True
        else:
            return False

    
    #Methode zum Ausführem des K-Means Clustering
    def run_manual_k(self, cluster_amount, data):
        
        d = data
        cluster_dict = self.introduce_cluster(cluster_amount, d)
        cluster_dict_before = {cluster: (0,0) for cluster in list(cluster_dict.keys())}
        iteration_count = 0
        progress_series = pd.Series([(round(coordinate[0],2), round(coordinate[1],2)) for coordinate in cluster_dict.values()], index = cluster_dict.keys())
        progress_frame = pd.DataFrame([progress_series])
        while not(self.check_cluster_change(cluster_dict_before, cluster_dict, 0.1)):
            d = self.calculate_distances(cluster_dict, d)
            d = self.assign_clusters(cluster_dict, d)
            cluster_dict_before = copy.deepcopy(cluster_dict)
            cluster_dict = self.centroid_coordinates(cluster_dict, d)
            iteration_count += 1
            progress_series = pd.Series([(round(coordinate[0],2), round(coordinate[1], 2)) for coordinate in cluster_dict.values()], index = cluster_dict.keys())
            progress_frame = progress_frame.append(progress_series, ignore_index=True)
        
        return d                            # <- Auskommentieren, um die Cluster zu plotten
                                            #Die Ausgabe wurde ursprünglich für streamlit designed und wird ohne das Paket nicht funktionieren
                                            #Die Clusteranzahl ist auf 8 begrenzt, um das Farbspektrum nutzen zu können
        
        progress_frame = progress_frame[:-1]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        cluster_colors = {cluster: colors[i] for i, cluster in enumerate(cluster_dict.keys())}
            
        for i, cluster in enumerate(list(cluster_dict.keys())):
            plt.scatter(d.loc[d["assigned to"] == cluster]["x"], d.loc[d["assigned to"] == cluster]["y"], c= clr.to_rgba(cluster_colors[cluster], 0.3))
        
        cluster_points=[(coordinate[0], coordinate[1]) for cluster, coordinate in cluster_dict.items()]
        cluster_frame = pd.DataFrame(cluster_points, index=list(cluster_dict.keys()))
       
        cluster_frame.apply(lambda row: plt.scatter(row[0], row[1], c= cluster_colors[row.name]), axis = 1)
        #st.pyplot()    #Auskommentieren um mit streamlit zu plotten
     
    

def main():
    km = KMeans()
    print(km.buildData())

if __name__ == "__main__":
    main()