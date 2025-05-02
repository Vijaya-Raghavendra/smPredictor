import plotly.graph_objects as go
import streamlit as st
from src.dataHandler import DataLoaderAndSaver
from src.model import StockPredictor
import altair as alt
import time


st.set_page_config(page_title="smPredcitor")
st.title("Stock Market Predictor")
dataloader = DataLoaderAndSaver()
lst = [ele[:-3] for ele in dataloader.loadList()]


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Work+Sans&display=swap');

    .worksans {
        font-family: 'Work Sans', sans-serif;
    }
    </style>

    <div class='worksans'>
        <h3 style='margin-bottom: 0;'>1. Fetch the Current Stock Price Data</h3>
        <div style='font-size: 16px;'>
            This step downloads and processes stock market data at two granularities:
            <ul style='margin-bottom: -0.3em;'>
                <li><b>Daily data</b> for the past <code>5 years</code></li>
                <li><b>5-minute interval data</b> for the past <code>1 month</code></li>
            </ul>
            The data is cleaned, saved locally, and prepared for model training or analysis.<br><br>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if st.button("Fetch and Process Stock Data"):
    dataloader.fetchAndSaveData("1d", "5y")
    dataloader.processAndSaveData("1d")
    dataloader.fetchAndSaveData("5m", "1mo")
    dataloader.processAndSaveData("5m")


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Work+Sans&display=swap');

    .worksans {
        font-family: 'Work Sans', sans-serif;
    }
    </style>

    <div class='worksans'>
        <h3 style='margin-bottom: 0;'>2. Choose a Stock for Continuous Prediction</h3>
        <div style='font-size: 18px;'>
            In this step, you select a stock symbol for which the model will start predicting future prices in real-time. 
            Upon pressing the start button, the system continuously fetches the latest data, makes a prediction for the next point, 
            compares it with the actual value (when available), and updates the model accordingly to improve future forecasts.<br><br>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<h1 style='font-size: 23px; font-weight:500; margin-bottom: -2em; margin-top: -1em;'>Select a stock 📈:</h1>",
    unsafe_allow_html=True,
)
selectedStock = st.selectbox("", lst)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


dataframe = dataloader.getProcessedData(selectedStock, "5m")
dataframe.set_index("Datetime", inplace=True)


# ---------- Initial Graph----------
df = dataframe[["Close"]][-100:].copy()
df["Time Period"] = range(len(df))

y_min, y_max = df["Close"].min(), df["Close"].max()
buffer = 0.05 * (y_max - y_min)

chart = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x="Time Period:Q",
        y=alt.Y("Close:Q", scale=alt.Scale(domain=[y_min - buffer, y_max + buffer])),
    )
)
st.altair_chart(chart, use_container_width=True)
# ----------------------------------


st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Work+Sans&display=swap');

    .worksans {
        font-family: 'Work Sans', sans-serif;
    }
    </style>

    <div class='worksans'>
        <h3 style='margin-bottom: 0;'>3. Start the Prediction</h3>
        <div style='font-size: 18px;'>
            In this step, you initiate the continuous prediction process for the selected stock. 
            When you press the "Start Prediction" button, the model begins predicting future stock prices 
            at regular intervals. The system fetches the most recent data, makes real-time predictions, 
            and continuously updates the model with new data points, improving its forecasting accuracy.<br><br>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if st.button("🔁 Start Prediction"):

    model = StockPredictor()
    x, y, df = model.prepareData(
        ["Open", "Close", "Volume", "High", "Low"], 5, "Close", selectedStock, "5m"
    )

    split_idx = int(0.9 * len(x))

    xTrain = x[:split_idx]
    yTrain = y[:split_idx]
    xTest = x[split_idx:]
    yTest = y[split_idx:]

    maeLoss, mseLoss = model.trainModel(st, xTrain, yTrain, 3)

    # History for plotting
    history = df["Close"].tolist()
    historical_line = history[split_idx - 50 : split_idx]
    true_line = history[split_idx - 50 : split_idx].copy()
    pred_line = [None] * 50


    # Plot loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=maeLoss, mode="lines", name="MAE"))
    fig.update_layout(
        title="MAE Loss Over Time",
        xaxis_title="Epoch",
        yaxis_title="MAE",
        template="plotly_dark"  # Optional: dark theme
    )
    st.plotly_chart(fig, use_container_width=True)

    # Final prediction vs true
    st.markdown(
        f"""
        <div style="width: 100%;">
            <table style="width: 100%; text-align: left;">
                <tr><th style="background: rgba(255, 255, 255, 0.1)">Metric</th>
                    <th style="background: rgba(255, 255, 255, 0.1)">Value</th>
                </tr>
                <tr><td>MSE</td><td>{round(mseLoss[-1], 2)}</td></tr>
                <tr><td>MAE</td><td>{round(maeLoss[-1], 2)}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


    placeholder_chart = st.empty()
    placeholder_text = st.empty()
    rows = []
    for i, (x_i, y_i) in enumerate(zip(xTest, yTest)):
        pred = model.model.predict_one(x_i)
        pred_line.append(pred)
        true_line.append(y_i)
        rows.append(f"""
            <tr>
                <td style="padding: 8px;">{i + 1}</td>
                <td style="padding: 8px;">{round(pred, 2)}</td>
                <td style="padding: 8px;">{round(y_i, 2)}</td>
            </tr>
        """)

        table_html = f"""
            <table style="width: 100%; text-align: left; border-collapse: collapse;">
                <tr>
                    <th style="padding: 8px; background-color: rgba(255, 255, 255, 0.1);">Prediction #</th>
                    <th style="padding: 8px; background-color: rgba(255, 255, 255, 0.1);">Predicted</th>
                    <th style="padding: 8px; background-color: rgba(255, 255, 255, 0.1);">Actual</th>
                </tr>
                {''.join(rows)}
            </table>
        """

        placeholder_text.markdown(table_html, unsafe_allow_html=True)

        # chart plotting remains unchanged


        # Then show the updated chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=true_line, name="True", line=dict(color="green", width=2))
        )
        fig.add_trace(
            go.Scatter(
                y=pred_line,
                name="Predicted",
                mode="lines+markers",
                line=dict(color="orange", width=2),
                marker=dict(symbol="circle", size=6),
            )
        )
        placeholder_chart.plotly_chart(fig, use_container_width=True)

        model.model.learn_one(x_i, y_i)
        time.sleep(10)
