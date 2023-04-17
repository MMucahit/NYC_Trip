import streamlit as st
import pandas as pd

import xgboost as xgb

import pickle

## Load Model
def loadModel():
    with open('./models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

## Prepare İnput
def prepareData(pu, do, trip_distance):
    with open('./models/preprocessor.pkl', 'rb') as f:
        dv = pickle.load(f)

    data = dv.transform({'PU_DO': str(pu) + '_' + str(do), 'trip_distance':trip_distance})
    return xgb.DMatrix(data)

def calculateTripDistance(df, pu, do):
    trip_distance = df[(df['PULocationID'] == pu) & (df['DOLocationID'] == do)]
    return trip_distance['trip_distance'].mean()

df_PU = pd.read_csv("./data/PU.csv")
df_DO = pd.read_csv("./data/DO.csv")

df = pd.read_parquet("./data/green_tripdata_2022-02.parquet")

pu = st.selectbox(
    'Select Pickup Location',
    df_PU['PULocationName'])

do = st.selectbox(
    'Select Dropoff Location',
    df_DO['DOLocationName'])

trip_distance = calculateTripDistance(df, pu, do)

pu_value = df_PU[df_PU['PULocationName'] == pu]['PULocationID'].iloc[0]
do_value = df_DO[df_DO['DOLocationName'] == do]['DOLocationID'].iloc[0]

data = prepareData(pu_value, do_value, trip_distance)
predict = loadModel().predict(data)

if st.button('Predict Duration'):   
    st.write('Duration:', f"{int(predict[0])} minutes")