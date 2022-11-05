import streamlit as st
import pandas as pd
from plotHelper import plot_expansion_graph, plot_feature_categories, plot_hedge_size, plot_imbalance, plot_sampling

DATA_URL = "auc_all.csv"
df = pd.read_csv(DATA_URL)

plot_options = [
        "Mean - N expansion",
        "Mean - Features(Category)",
        "Mean - Hyper-edge Size",
        "Mean - Imbalance Ratio",
        "Mean - Negative Sampling"
    ]

with st.sidebar:
    page = st.selectbox("Choose Plot to view", plot_options)

if page == plot_options[0]:
    plot_expansion_graph(df)
elif page == plot_options[1]:
    plot_feature_categories(df)
elif page == plot_options[2]:
    plot_hedge_size(df)
elif page == plot_options[3]:
    plot_imbalance(df)
elif page == plot_options[4]:
    plot_sampling(df)
