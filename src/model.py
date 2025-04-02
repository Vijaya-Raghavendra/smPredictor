from deep_river.regression import RollingRegressorInitialized
from src.lstmModule import LstmModule
from river import metrics
import pandas as pd
import numpy as np


class StockPredictor:
    """
    A class that wraps the LSTM model inside rolling regressor of deep-river
    library and handles the training and predicting functionalities of the model.
    """

    def __init__(self, n_features):
        # self.model = RollingRegressorInitialized(
        #     module=LstmModule(n_features, 16),
        #     loss_fn="mse",
        #     optimizer_fn="adam",
        #     window_size=20,
        #     lr=0.001,
        #     append_predict=True,
        # )
        # self.metric = metrics.MAE()
        pass

    
    def prepareData(
        self, dataframe: pd.DataFrame, toLagFeatures: list, lag: int, predictionVarList: str
    ):
        # df = dataframe.copy()
        # # creating the columns for the lag features
        # for targetVar in toLagFeatures:
        #     for i in range(1, lag + 1):
        #         df[f"{targetVar}_{i}"] = df[targetVar].shift(i)
        #     df.dropna(inplace=True)

        # # ------------------ change this -----------------------
        # x = df.drop(columns=predictionVarList, axis=1)
        # y = df["Close"]
        # return x.to_dict(orient="records"), y.to_list()
        pass

    def trainModel(self, x, y, epochs=10):
        # for epoch in range(epochs):
        #     epoch_loss = 0
        #     for i in range(len(y)):
        #         y_pred = self.model.predict_one(x[i])
        #         loss = (y_pred - y[i]) ** 2
        #         epoch_loss += loss
        #         self.metric.update(y_true=y[i], y_pred=y_pred)
        #         self.model.learn_one(x[i], y[i])

        #     avg_loss = epoch_loss / len(y)
        #     print(
        #         f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MAE: {self.metric.get():.4f}"
        #     )
        pass

    def evaluate(self, X, y):
        # y_pred = [self.model.predict_one(xi) for xi in X]
        # mse = np.mean((np.array(y) - np.array(y_pred)) ** 2)
        # return mse
        pass