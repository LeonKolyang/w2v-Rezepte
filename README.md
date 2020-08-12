# W2V-Rezepte
Deployed at:
www.w2v-rezepte.de

Frontend for the W2V-Rezepte Tool using the machine learning microservices under
https://github.com/LeonKolyang/w2v-microservices


Companion project for the bachelorthesis "Data Mining: Extraction of Online Recipes and Processing with Word2Vec". The goal of the bachelorthesis was the preparation and definition of a machine learning modell, to recognize characteristic elements of online recipes. A further description of the underlying idea and concept can be found on the overview page of the project.

This project implements the methods researched during the bachelorthesis in an enclosed software architecture. All steps of the major machine modelling tasks (acquisition and preprocessing of data, training and optimization of the model, use of the trained model) are build as independent microservices. The preprocessing, training and optimization tasks are displayed in detail and seperate from the use of the trained model.

As this projects reflects ongoing work, some parts might not be included yet. Application performance has a minor priority, this might lead to some delays.