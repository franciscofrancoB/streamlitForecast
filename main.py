import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(
    page_title="Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def style():
    st.markdown(
        """
        <style>
        :root{
            --bg-main:#FFFFFF;          /* main background */
            --bg-sidebar:#FFFF00;       /* sidebar background */
            --text-main:#36454F;        /* titles text*/
        }
        /* layout */
        html,body,.stApp{background-color:var(--bg-main);color:var(--text-main);font-family:'Inter',sans-serif;}
        div[data-testid='stHeader']{background:linear-gradient(90deg,var(--bg-sidebar) 0%,var(--bg-card) 100%);height:3rem;}
        /* sidebar */
        .stSidebar{background-color:var(--bg-sidebar);}
        .stSidebar h1,.stSidebar h2,.stSidebar h3,.stSidebar h4,.stSidebar h5,.stSidebar h6{color:var(--secondary);}
        /* titles */
        h1,h2,h3,h4,h5,h6{color:var(--secondary);font-weight:700;}
        /* buttons */
        .stButton>button{background-color:var(--primary);color:var(--text-main);border:none;border-radius:8px;padding:0.6rem 1.2rem;font-weight:600;}
        .stButton>button:hover{background-color:var(--accent);}
        /* number input / selectbox */
        .stNumberInput>div>div>input,
        .stSelectbox>div>div{background-color:var(--bg-input);border:1px solid var(--primary);border-radius:8px;color:var(--text-main);}
        /* file uploader */
        .stFileUploader>div>div{background-color:var(--bg-card);border:2px dashed var(--primary);border-radius:10px;color:var(--text-main);}
        .stFileUploader span{color:var(--text-main) !important;font-weight:600;}
        .stFileUploader>div>div:hover{border-color:var(--accent);}
        /* data frame */
        .stDataFrame{background-color:var(--bg-card);border-radius:10px;box-shadow:0 2px 4px rgba(0,0,0,0.4);}
        /* plotly bg */
        .js-plotly-plot .plotly{background-color:var(--bg-card);}
        </style>
        """,
        unsafe_allow_html=True,
    )

def detect_date_column(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            pd.to_datetime(df[col])
            return col
        except Exception:
            pass
    return None

def prepare_data(df, date_col, y_col):
    df = df[[date_col, y_col]].dropna()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "ds", y_col: "y"}).sort_values("ds")
    return df

def infer_frequency(series):
    freq = pd.infer_freq(series.sort_values())
    if freq is None:
        return "D"
    if freq.endswith("M") and not freq.endswith("MS"):
        return "MS"
    return freq

def forecast_prophet(df, periods, freq):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def plot_series(history, forecast, horizon):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            mode="lines",
            line=dict(color="#158FD1", width=2.5),
            name="Historical Data",
        )
    )
    fc_tail = forecast.tail(horizon)
    fig.add_trace(
        go.Scatter(
            x=fc_tail["ds"],
            y=fc_tail["yhat"],
            mode="lines",
            line=dict(color="#D4AF37", width=3),
            name="Forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc_tail["ds"]) + list(fc_tail["ds"][::-1]),
            y=list(fc_tail["yhat_upper"]) + list(fc_tail["yhat_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(142,68,173,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Uncertainty",
        )
    )
    window_size = max(horizon * 3, int(len(history) * 0.2))
    start_index = max(len(history) - window_size, 0)
    fig.update_xaxes(range=[history["ds"].iloc[start_index], forecast["ds"].iloc[-1]])
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="#F5F5F5",
        plot_bgcolor="#F5F5F5",
        font=dict(color="#36454F"),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

def app():
    style()
    st.title("Forecasting Platform")
    with st.sidebar:
        file = st.file_uploader("Upload CSV", type=["csv"])
        horizon = st.number_input("Forecast horizon (future periods)", 1, 500, 12)
    if file:
        df = pd.read_csv(file)
        date_col = detect_date_column(df)
        if date_col is None:
            st.error("No date column detected.")
            return
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("No numeric column detected.")
            return
        y_col = numeric_cols[0] if len(numeric_cols) == 1 else st.selectbox("Choose value column", numeric_cols)
        st.subheader("Data preview")
        st.dataframe(df.head(), use_container_width=True)
        prepared = prepare_data(df, date_col, y_col)
        freq = infer_frequency(prepared["ds"])
        st.caption(f"Detected data frequency: {freq}")
        forecast_df = forecast_prophet(prepared, horizon, freq)
        st.subheader("Forecast")
        plot_series(prepared, forecast_df, horizon)
        csv = forecast_df.tail(horizon).to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", csv, file_name="forecast.csv", mime="text/csv")
    else:
        st.info("Awaiting CSV upload.")

if __name__ == "__main__":
    app()
