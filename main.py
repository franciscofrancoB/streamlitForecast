import streamlit as st
import pandas as pd
import numpy as np

def plot_dataframe(df):
    df_copy = df.copy()
    date_cols = [c for c in df_copy.columns if pd.api.types.is_datetime64_any_dtype(df_copy[c])]
    if not date_cols:
        for c in df_copy.columns:
            try:
                df_copy[c] = pd.to_datetime(df_copy[c])
                date_cols.append(c)
                break
            except Exception:
                continue
    if date_cols:
        dt_col = date_cols[0]
        df_copy.set_index(dt_col, inplace=True)
    st.line_chart(df_copy.select_dtypes(include=[np.number]))

def main():
    st.title("Sales Forecasting")
    uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        #st.subheader("Data Preview")
        #st.dataframe(df)
        st.subheader("Data Plot")
        plot_dataframe(df)
    else:
        st.info("Awaiting CSV file to be uploaded.")

if __name__ == "__main__":
    main()

    