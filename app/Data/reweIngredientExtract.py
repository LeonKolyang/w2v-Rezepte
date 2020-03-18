#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pprint import pprint
import pandas as pd
from pymongo import MongoClient
client = MongoClient('localhost:27017')
db = client.Recipes.ckRecipes


# In[2]:


pipeline = [
    {"$project":{ 
      "_id":0,
      "ingredients":1}}]
ingredientList = list(db.aggregate(pipeline))


# In[3]:


columns = ["Description","Amount", "Unit" ]
#df= pd.DataFrame(columns=columns)
df= pd.DataFrame()
df = pd.DataFrame(ingredientList[0]["ingredients"])
df


# In[4]:


for ingredient in ingredientList[1:]:
    dfIn=pd.DataFrame(ingredient["ingredients"])
    df=df.append(dfIn,ignore_index=True)


# In[5]:


df


# In[6]:


df.to_csv('ckIngredients.csv', index=False)  

