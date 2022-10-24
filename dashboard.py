import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

HEDGE_SIZES = [4, 5]

DATA_URL = "auc_all.csv"
df = pd.read_csv(DATA_URL)


def process_selection(data, hedges=None, samples=None, imbs=None, features=None):
    if hedges is not None:
        data = data[
            data['hedge'].isin(hedges)
        ]
    if samples is not None:
        data = data[
            data['neg_sample'].isin(samples)
        ]
    if imbs is not None:
        data = data[
            data['imbalance'].isin(imbs)
        ]
    if features is not None:
        data = data[
            data['feature'].isin(features)
        ]
    return data


def data_filter_selection(hedge=True, sample=True, imb=True, feature=True):
    hedge_selection = None
    sample_selection = None
    imb_selection = None
    feature_selection = None

    if hedge:
        hedge_options = [4, 5]
        hedge_selection = st.sidebar.multiselect(
            'Select hyper edge size/s', hedge_options, key='2',
            default=hedge_options
        )
    if sample:
        sample_types = ["clique", "star", "tailed"]
        sample_selection = st.sidebar.multiselect(
            'Select negative sampling type/s', sample_types, key='3',
            default=sample_types
        )
    if imb:
        imb_options = [1, 2, 5, 10]
        imb_selection = st.sidebar.multiselect(
            'Select imbalance ratio/s', imb_options, key='4',
            default=imb_options
        )
    if feature:
        feature_options = ["aa", "am", "cn", "gm", "hm", "jc"]
        feature_selection = st.sidebar.multiselect(
            'Select feature/s', feature_options, key='5',
            default=feature_options
        )
    return hedge_selection, sample_selection, imb_selection, feature_selection


def cols_from_setting(data):
    data["hedge"] = data.setting.apply(lambda x: int(x.split('_')[0].split('-')[-1]))
    data["neg_sample"] = data.setting.apply(lambda x: x.split('_')[1].split('-')[1])
    data["imbalance"] = data.setting.apply(lambda x: int(x.split('_')[1].split('-')[-1]))
    data["feature"] = data.setting.apply(lambda x: x.split('_')[-1])
    data.drop(columns=["setting"] + list(data.filter(regex='ROC')), inplace=True)
    return data


def plot_expansion_graph(data):
    st.title("Mean AUC-PR by n-order expansion")
    data = cols_from_setting(data)

    hedge_selection, sample_selection, imb_selection, feature_selection = data_filter_selection()
    info_text = "Negative sampling selections: {}\n\n" \
                "Imbalance ratio selections: {}\n\n" \
                "Feature selections: {}\n\n" \
                "Hyper edge selections: {}"

    info_text = info_text.format(
            ", ".join(sample_selection),
            ", ".join(map(str, imb_selection)),
            ", ".join(feature_selection),
            ", ".join(map(str, hedge_selection))
        )

    st.info(info_text)

    data = process_selection(
        data,
        hedges=hedge_selection,
        samples=sample_selection,
        features=feature_selection,
        imbs=imb_selection
    )

    data = data[['Graph Name'] + list(df.filter(regex='PR'))]
    data = data.groupby(by='Graph Name', as_index=False).mean()
    data = data.rename(
        {"PR1": "2", "PR2": "3", "PR3": "4"}, axis="columns"
    )

    data_t = data.drop(columns=['Graph Name']).transpose()
    data_t.columns = data['Graph Name']
    st.dataframe(data_t)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data_t.index, y=data_t['email-Enron'], name="email-Enron",
            line={'width': 7}, marker={'size': 18}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data_t.index, y=data_t['email-Eu'], name="email-EU",
            line={'width': 7}, marker={'size': 18}
        )
    )
    fig.update_layout(
        title="The mean AUC-PR for different n-order expansions",
        yaxis=dict(
            title_text="AUC-PR",
            titlefont=dict(size=24)
        ),
        xaxis=dict(
            title_text="n-order expansion",
            titlefont=dict(size=24)
        ),
        titlefont=dict(size=24),
        font=dict(
            size=24
        )
    )
    st.plotly_chart(fig)


def plot_feature_categories(data):
    st.title("Mean AUC-PR by different feature categories")

    data = cols_from_setting(data)
    hedge_selector, sample_selector, imb_selector, _ = data_filter_selection(feature=False)

    data = process_selection(
        data,
        hedges=hedge_selector,
        samples=sample_selector,
        imbs=imb_selector
    )

    info_text = "Negative sampling selections: {}\n\nImbalance ratio selections: {}\n\nHyper edge Selections: {}".format(
        ", ".join(sample_selector),
        ", ".join(map(str, imb_selector)),
        ", ".join(map(str, hedge_selector))
    )
    st.info(info_text)

    # GM, GM, AM
    mdf = data[data['feature'].isin(['gm', 'hm', 'am'])]
    mdf = mdf[['Graph Name'] + list(mdf.filter(regex='PR'))]
    mdf = mdf.groupby(by=['Graph Name'], as_index=False).mean()
    mdf["feature"] = "Mean"
    mdf["Avg"] = mdf[list(mdf.filter(regex='PR'))].mean(axis=1)
    mdf.drop(columns=list(mdf.filter(regex='PR')), inplace=True)
    # CN, JC, AA
    mdn = data[data['feature'].isin(['cn', 'jc', 'aa'])]
    mdn = mdn[['Graph Name'] + list(mdn.filter(regex='PR'))]
    mdn = mdn.groupby(by=['Graph Name'], as_index=False).mean()
    mdn["feature"] = "Neighbor"
    mdn["Avg"] = mdn[list(mdn.filter(regex='PR'))].mean(axis=1)
    mdn.drop(columns=list(mdn.filter(regex='PR')), inplace=True)
    target_df = pd.concat([mdf, mdn])
    del (mdf, mdn)

    f1 = target_df[target_df['Graph Name'] == 'email-Enron']
    f2 = target_df[target_df['Graph Name'] == 'email-Eu']
    fList = target_df['feature'].unique().tolist()

    col1, col2 = st.columns(2)
    col1.dataframe(f1)
    col2.dataframe(f2)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=fList, y=f1["Avg"], name="email-Enron")
    )
    fig.add_trace(
        go.Bar(x=fList, y=f2["Avg"], name="email-Eu")
    )

    fig.update_layout(
        title="The mean AUC-PR for different features",
        yaxis=dict(
            title_text="AUC-PR",
            titlefont=dict(size=24)
        ),
        xaxis=dict(
            title_text="Features",
            titlefont=dict(size=24)
        ),
        titlefont=dict(size=24),
        font=dict(
            size=24
        ),
        yaxis_range=[0, 1],
    )

    st.plotly_chart(fig)
    st.info(
        "Mean features are (GM, HM, AM)\n\n"
        "Neighbor features are (CN, JC, AA)"
        )


def plot_hedge_size(data):
    st.title("Mean AUC-PR for hyper edge size")
    _, sample_selection, imb_selection, feature_selection = data_filter_selection(hedge=False)

    data = cols_from_setting(data)

    info_text = "Negative sampling selections: {}\n\nImbalance ratio selections: {}\n\nFeature Selections: {}".format(
        ", ".join(sample_selection),
        ", ".join(map(str, imb_selection)),
        ", ".join(feature_selection)
    )
    st.info(info_text)

    data = process_selection(data, samples=sample_selection, imbs=imb_selection, features=feature_selection)

    data = data[["Graph Name", "hedge"] + list(data.filter(regex="PR"))]
    data = data.groupby(by=["Graph Name", "hedge"], as_index=False).mean()
    data['Avg'] = data[list(data.filter(regex='PR'))].mean(axis=1)
    data.drop(columns=list(data.filter(regex='PR')), inplace=True)

    f1 = data[data['Graph Name'] == 'email-Enron']
    f2 = data[data['Graph Name'] == 'email-Eu']
    fList = data['hedge'].unique().tolist()

    col1, col2 = st.columns(2)
    col1.dataframe(f1)
    col2.dataframe(f2)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=fList, y=f1["Avg"], name="email-Enron")
    )
    fig.add_trace(
        go.Bar(x=fList, y=f2["Avg"], name="email-Eu")
    )

    fig.update_layout(
        title="The mean AUC-PR for different hyper edge sizes",
        yaxis=dict(
            title_text="AUC-PR",
            titlefont=dict(size=24)
        ),
        xaxis=dict(
            title_text="Hyperedge Size",
            titlefont=dict(size=24)
        ),
        titlefont=dict(size=24),
        font=dict(
            size=24
        ),
        yaxis_range=[0, 1],
    )

    st.plotly_chart(fig)


def plot_imbalance(data):
    st.title("Mean AUC values for different imbalance ratios")
    data = cols_from_setting(data)
    hedge_selection, sample_selection, _, feature_selection = data_filter_selection(imb=False)

    info_text = "Negative sampling selections: {}\n\nHyper edge selections: {}\n\nFeature Selections: {}".format(
        ", ".join(sample_selection),
        ", ".join(map(str, hedge_selection)),
        ", ".join(feature_selection)
    )
    st.info(info_text)
    data = process_selection(data, samples=sample_selection, hedges=hedge_selection, features=feature_selection)

    data = data[["Graph Name", "imbalance"] + list(data.filter(regex="PR"))]
    data = data.groupby(by=["Graph Name", "imbalance"], as_index=False).mean()
    data['Avg'] = data[list(data.filter(regex="PR"))].mean(axis=1)
    data.drop(columns=list(data.filter(regex="PR")), inplace=True)
    f1 = data[data['Graph Name'] == "email-Enron"]
    f2 = data[data['Graph Name'] == "email-Eu"]
    xNames = data['imbalance'].unique().tolist()

    col1, col2 = st.columns(2)
    col1.dataframe(f1)
    col2.dataframe(f2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xNames, y=f1['Avg'], name="email-Enron",
            line={'width': 7}, marker={'size': 18}
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xNames, y=f2['Avg'], name="email-Eu",
            line={'width': 7}, marker={'size': 18, 'symbol': 'diamond'}
        )
    )

    fig.update_layout(
        title="Mean AUC values for imbalance values",
        yaxis=dict(
            title_text="AUC-PR",
            titlefont=dict(size=24)
        ),
        xaxis=dict(
            title_text="Imbalance ratios",
            titlefont=dict(size=24)
        ),
        titlefont=dict(size=24),
        font=dict(
            size=24,
        ),

    )
    st.plotly_chart(fig)
    return


def plot_sampling(data):
    st.title("Mean AUC values for different negative sampling techniques")
    data = cols_from_setting(data)
    hedge_selection, _, imb_selection, feature_selection = data_filter_selection(sample=False)
    info_text = "Hyper edge selections: {}\n\nImbalance ratio selections: {}\n\nFeature Selections: {}".format(
        ", ".join(map(str, hedge_selection)),
        ", ".join(map(str, imb_selection)),
        ", ".join(feature_selection)
    )
    st.info(info_text)
    data = process_selection(data, imbs=imb_selection, hedges=hedge_selection, features=feature_selection)
    data = data[["Graph Name", "neg_sample"] + list(data.filter(regex="PR"))]
    data = data.groupby(by=["Graph Name", "neg_sample"], as_index=False).mean()
    data['Avg'] = data[list(data.filter(regex="PR"))].mean(axis=1)
    data = data.drop(columns=list(data.filter(regex='PR')))

    f1 = data[data["Graph Name"] == "email-Enron"]
    f2 = data[data["Graph Name"] == "email-Eu"]
    xnames = data.neg_sample.unique().tolist()

    cols = st.columns(2)

    cols[0].dataframe(f1)
    cols[1].dataframe(f2)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=xnames,
            y=f1['Avg'],
            name="email-Enron"
        )
    )

    fig.add_trace(
        go.Bar(
            x=xnames,
            y=f2['Avg'],
            name="email-Eu"
        )
    )

    fig.update_layout(
        title="The mean AUC-PR for different negative sampling techniques",
        yaxis=dict(
            title_text="AUC-PR",
            titlefont=dict(size=24)
        ),
        xaxis=dict(
            title_text="Negative Sampling",
            titlefont=dict(size=24)
        ),
        titlefont=dict(size=24),
        font=dict(
            size=24
        ),
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig)

    return


st.title("Evaluation for Tailed Negative Sampling")
st.sidebar.title("Dashboard")

plot_options = [
    "Mean - N expansion",
    "Mean - Features(Category)",
    "Mean - Hyper-edge Size",
    "Mean - Imbalance Ratio",
    "Mean - Negative Sampling"
]
select = st.sidebar.selectbox('Plot option', plot_options, key='1')


if select == plot_options[0]:
    plot_expansion_graph(df)
elif select == plot_options[1]:
    plot_feature_categories(df)
elif select == plot_options[2]:
    plot_hedge_size(df)
elif select == plot_options[3]:
    plot_imbalance(df)
elif select == plot_options[4]:
    plot_sampling(df)
