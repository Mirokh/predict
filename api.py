#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import get_ipython
import json
import datetime
import pyspark
from datetime import timedelta,date
import pyspark
import pyspark.sql.functions as func
from pyspark.sql.window import *
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import sys


# In[2]:


def apply_smooth_mean(encoded_list,dt,path):
  """
    applies smooth mean for a given columns

    :param dt: dt for which applly smooth mean
    :param encoded_list: list of columns
    :return: returns df with encoded columns
    """
  for i in encoded_list:
    smooth = spark.read.csv(path + i,header=True)
    
    dt = dt.join(smooth,[i],'left').\
    drop(i,'mean_smooth').\
    withColumnRenamed('smooth',i).\
    withColumn(i,col(i).cast(FloatType()))
  return dt


# In[3]:


encoding_list = ['PreferredLoginDevice',
                 'PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']


# In[4]:


spark = SparkSession.\
builder.\
appName('MyApp').\
config("spark.driver.extraJavaOptions", "-Djline.terminal.width=4000").\
getOrCreate()


# In[5]:


# Import necessary libraries
import pandas as pd
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify

# Load the XGBoost model
with open('/Users/hov.odn/noor_games/xgb_model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

# Create a Flask app
app = Flask(__name__)

@app.route('/helth')

def helth():
    return 'hello world'

# Define the API endpoint
@app.route('/predictions', methods=['POST'])

def predict():
    # Get the JSON data from the POST request
    data = request.data
    dt = pd.read_json(data)[loaded_model.feature_names_in_]
    dt = apply_smooth_mean(encoding_list,spark.createDataFrame(dt),'/Users/hov.odn/noor_games/encodings_').toPandas()
    
    # Use the XGBoost model to make predictions on the data
    predictions = loaded_model.predict(dt)

   # Convert the predictions to a JSON object and return it
    return jsonify(predictions=predictions.tolist())

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




