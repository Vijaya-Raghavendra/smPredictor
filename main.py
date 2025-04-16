import pandas as pd
import streamlit as st
from src.dataHandler import DataLoaderAndSaver
from src.model import StockPredictor

st.set_page_config(page_title="smPredcitor")
st.title("Stock Market Predictor")

dataloader = DataLoaderAndSaver()

if st.button("Fetch Data"):
    dataloader.fetchAndSaveData("1d", "5y")
    dataloader.processAndSaveData("1d")
    dataloader.fetchAndSaveData("5m", "1mo")
    dataloader.processAndSaveData("5m")


lst = [ele[:-3] for ele in dataloader.loadList()]
selectedStock = st.selectbox("Select a stock", lst)

dataframe = dataloader.getProcessedData(selectedStock, "5m")
dataframe.set_index("Datetime", inplace=True)

st.line_chart(dataframe[["Close"]][-100:])

if st.button("Train the model"):
    model = StockPredictor()
    xTrain, yTrain, xTest, yTest = model.prepareData(["Open", "Close", "Volume", "High", "Low"], 5, "Close", selectedStock, "5m")
    errors = model.trainModel(st, xTrain, yTrain, 30)
    prediction = model.predictNext(xTrain)
    st.write(prediction)
    st.line_chart(errors)
