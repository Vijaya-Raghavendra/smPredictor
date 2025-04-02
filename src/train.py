from src.model import StockPredictor
from src.dataHandler import DataLoaderAndSaver
import torch


# class ModelTrainer:

#     def __init__(self, ticker, interval):
#         self.ticker = ticker
#         self.interval = interval
#         self.dataloader = DataLoaderAndSaver()
#         self.tickerList = [ele[:-3] for ele in self.dataloader.loadList()]

#     def trainModel(self):

#         if self.interval == "5m":
#             columnsToDrop = ["Open", "Volume", "Datetime"]
#         else:
#             columnsToDrop = ["Open", "Volume", "Date"]

#         # getting the data
#         dataframe = self.dataloader.getProcessedData(self.ticker, self.interval).drop(
#             columns=columnsToDrop
#         )

#         # initiating the model
#         self.model = StockPredictor(n_features=5)

#         # preparing the data
#         x, y = self.model.prepareData(dataframe, "Close", 5)
#         split = int(0.9 * len(x))
#         xTrain, yTrain = x[:split], y[:split]
#         xTest, yTest = x[split:], y[split:]

#         self.model.trainModel(xTrain, yTrain, epochs=10)
#         mse = self.model.evaluate(xTest, yTest)
#         print(f"MAE: {self.model.metric.get():.4f}, MSE: {mse:.4f}")

#         return
