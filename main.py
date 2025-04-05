from src.dataHandler import DataLoaderAndSaver
from src.model import StockPredictor

dataloader = DataLoaderAndSaver()

# fetching "1d" data for "5y"
dataloader.fetchAndSaveData("1d", "5y")
dataloader.processAndSaveData("1d")

# fetching "5m" data for "6m"
dataloader.fetchAndSaveData("5m", "1mo")
dataloader.processAndSaveData("5m")

model = StockPredictor()
xTrain, yTrain, xTest, yTest = model.prepareData(["Open", "Close", "Volume", "High", "Low"], 2, "Close", "ITC", "5m")
model.trainModel(xTrain, yTrain, 15)
model.evaluate(xTest, yTest)