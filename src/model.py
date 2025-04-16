from deep_river.regression import RollingRegressorInitialized
from src.dataHandler import DataLoaderAndSaver
from src.lstmModule import LstmModule
from river import metrics
import numpy as np


class StockPredictor:
    """
    A class that wraps the LSTM model inside rolling regressor of deep-river
    library and handles the training and predicting functionalities of the model.
    """

    def __init__(self):
        self.model = None
        self.dataloader = DataLoaderAndSaver()
        self.metric = metrics.MAE()
        return

    def standardize(self, series):
        mean = series.mean()
        std = series.std()
        if std != 0:
            series = (series - mean) / std
        else:
            series = 0
        return series

    def prepareData(
        self, lagList: list, lag: int, target: str, ticker: str, interval: str
    ):
        """
        Function to prepare the dataset to feed in the model. It also initializes the
        LSTM module used for prediction based on the number of features in the dataset.

        Returns: (xTrain, yTrain, xTest, yTest)
        """
        if interval == "5m":
            temp = "Datetime"
        else:
            temp = "Date"

        # load the data
        df = self.dataloader.getProcessedData(ticker, interval).drop(
            columns=[temp], axis=1
        )
        print(df)

        # creating lag features
        for targetVar in lagList:
            for i in range(1, lag + 1):
                df[f"{targetVar}_{i}"] = df[targetVar].shift(i)
            df.dropna(inplace=True)

        for column in df.columns:
            if column == "Close":
                continue
            df[column] = self.standardize(df[column])

        print(df)

        # preparing dataset
        x = df.drop(columns=[target] + ["Open"], axis=1).to_dict(
            orient="records"
        )  # remove "Open" from the features, cause its not available
        y = df["Close"].to_list()  # target

        # test-train split
        split = int(len(x) * 0.9)
        xTrain, yTrain = x[:split], y[:split]
        xTest, yTest = x[split:], y[split:]

        # initialize the model based on the number of features
        n_features = len(xTrain[0])
        self.initializeModel(n_features=n_features)

        return xTrain, yTrain, xTest, yTest

    def initializeModel(self, n_features):
        self.model = RollingRegressorInitialized(
            module=LstmModule(n_features, hidden_size=16),
            loss_fn="mse",
            optimizer_fn="adam",
            window_size=20,
            lr=0.001,
            append_predict=True,
        )
        return

    def trainModel(self, st, x, y, epochs=10):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(y)):
                y_pred = self.model.predict_one(x[i])
                loss = (y_pred - y[i]) ** 2
                epoch_loss += loss
                self.metric.update(y_true=y[i], y_pred=y_pred)
                self.model.learn_one(x[i], y[i])

            avg_loss = epoch_loss / len(y)
            losses.append(avg_loss)
            if st:  # Check if st is passed
                st.write(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MAE: {self.metric.get():.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, MAE: {self.metric.get():.4f}"
                )

        return losses

    def evaluate(self, xTest, yTest):
        """
        Evaluates the model based on the test data.
        Returns: MSE and MAE.
        """
        y_pred = [self.model.predict_one(xi) for xi in xTest]
        mse = np.mean((np.array(yTest) - np.array(y_pred)) ** 2)
        mae = self.metric.get()

        print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        return mse, mae

    def predictNext(self, xTrain):
        """
        Predicts the next value based on the most recent training data.
        Args:
            xTrain: The training data features used for prediction.
        Returns:
            predicted_value: The predicted next value for the target variable.
        """
        # Get the last data point from the training set
        lastDataPoint = xTrain[-1]
        print(lastDataPoint)
        # Predict the next value
        predicted_value = self.model.predict_one(lastDataPoint)
        return predicted_value
