from src.dataHandler import DataLoaderAndSaver
from src.model import StockPredictor


dataloader = DataLoaderAndSaver()

# fetching "1d" data for "5y"
# dataloader.fetchAndSaveData("1d", "5y")
# dataloader.processAndSaveData("1d")

# # fetching "5m" data for "6m"
# dataloader.fetchAndSaveData("5m", "1mo")
# dataloader.processAndSaveData("5m")

data = dataloader.getProcessedData("ITC", "1d").drop(columns=["Date"], axis=1)

s = StockPredictor(14)
s.prepareData(data, ["Open", "Close"], 5, "Close")