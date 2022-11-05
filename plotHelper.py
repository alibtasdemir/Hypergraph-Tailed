import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import *


def create_parameters_box(networks=None, hedges=None, sampling=None, imbs=None, features=None):
    info_text = ""
    if sampling:
        info_text += "Negative sampling selections: {}\n\n".format(", ".join(sampling))
    if networks:
        info_text += "Network selections: {}\n\n".format(", ".join(networks))
    if hedges:
        info_text += "Hyper edge selections: {}\n\n".format(", ".join(map(str, hedges)))
    if imbs:
        info_text += "Imbalance ratio selections: {}\n\n".format(", ".join(map(str, imbs)))
    if features:
        info_text += "Feature selections: {}\n\n".format(", ".join(features))

    with st.expander("Parameters"):
        st.info(info_text)


def process_selection(data, hedges=None, samples=None, imbs=None, features=None, networks=None):
    if networks is not None:
        data = data[
            data['Graph Name'].isin(networks)
        ]
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


def data_filter_selection(hedge=True, sample=True, imb=True, feature=True, network=True):
    hedge_selection = None
    sample_selection = None
    imb_selection = None
    feature_selection = None
    network_selection = None

    hedge_keys = [714, 1350, 2113, 2200, 2887, 3029, 3047, 4890, 5129, 5473, 5876, 5943, 6395, 7006, 8813, 9013, 9371,
                  9499, 9823, 9844]
    sample_keys = [463, 1042, 1851, 2253, 2780, 2976, 3559, 3731, 3747, 3754, 3857, 4723, 5821, 6007, 8581, 9094, 9184,
                   9230, 9296, 9657]
    imb_keys = [69, 719, 1039, 1498, 2411, 3210, 3412, 3683, 3932, 4623, 5032, 5263, 5684, 6015, 7353, 8168, 8622, 8625,
                8952, 9873]
    feature_keys = [541, 1004, 1338, 1521, 2023, 2087, 2367, 2503, 2505, 2705, 4265, 5335, 5758, 5816, 5885, 6145, 7749,
                    7901, 8602, 9088]
    network_keys = [1272, 1293, 2379, 2772, 2912, 4283, 4978, 5085, 6217, 6251, 6470, 6728, 6881, 7607, 8529, 8857,
                    8858, 8870, 9023, 9652]

    if network:
        network_options = GNS
        network_selection = st.sidebar.multiselect(
            'Select network/s', network_options, key=network_options[1],
            default=network_options
        )
        if len(network_selection) < 1:
            network_selection = None

    if hedge:
        st.sidebar.subheader("Select hyper edge size/s")
        hedge_options = HEDGE_SIZES
        hedge_dict = {x: st.sidebar.checkbox("{}".format(str(x)), value=True, key=key) for x, key in
                      zip(hedge_options, hedge_keys[:len(hedge_options)])}
        hedge_selection = [str(key) for key, value in hedge_dict.items() if value]
        if len(hedge_selection) < 1:
            hedge_selection = None
    if sample:
        st.sidebar.subheader("Select negative sampling type/s")
        sample_types = NEG_TYPES
        sample_dict = {x: st.sidebar.checkbox("{}".format(str(x)), value=True, key=key) for x, key in
                       zip(sample_types, sample_keys[:len(sample_types)])}
        sample_selection = [str(key) for key, value in sample_dict.items() if value]
        if len(sample_selection) < 1:
            sample_selection = None
    if imb:
        st.sidebar.subheader("Select imbalance ratio/s")
        imb_options = IMBS
        imb_dict = {x: st.sidebar.checkbox("{}".format(str(x)), value=True, key=key) for x, key in
                    zip(imb_options, imb_keys[:len(imb_options)])}
        imb_selection = [str(key) for key, value in imb_dict.items() if value]
        if len(imb_selection) < 1:
            imb_selection = None

    if feature:
        st.sidebar.subheader("Select feature/s")
        feature_options = FEATURES
        feature_dict = {x: st.sidebar.checkbox("{}".format(str(x)), value=True, key=key) for x, key in
                        zip(feature_options, feature_keys[:len(feature_options)])}
        feature_selection = [str(key) for key, value in feature_dict.items() if value]
        if len(feature_selection) < 1:
            feature_selection = None

    return hedge_selection, sample_selection, imb_selection, feature_selection, network_selection


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

    hedge_selection, sample_selection, imb_selection, feature_selection, network_selection = data_filter_selection()

    data = process_selection(
        data,
        hedges=hedge_selection,
        samples=sample_selection,
        features=feature_selection,
        imbs=imb_selection,
        networks=network_selection
    )

    data = data[['Graph Name'] + list(data.filter(regex='PR'))]
    data = data.groupby(by='Graph Name', as_index=False).mean()
    data = data.rename(
        {"PR1": "2", "PR2": "3", "PR3": "4"}, axis="columns"
    )

    data_t = data.drop(columns=['Graph Name']).transpose()
    data_t.columns = data['Graph Name']

    fig = go.Figure()

    for network in network_selection:
        fig.add_trace(
            go.Scatter(
                x=data_t.index, y=data_t[network], name=network,
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

    st.dataframe(data_t)

    create_parameters_box(
        networks=network_selection,
        hedges=hedge_selection,
        sampling=sample_selection,
        imbs=imb_selection,
        features=feature_selection
    )


def plot_feature_categories(data):
    st.title("Mean AUC-PR by different feature categories")

    data = cols_from_setting(data)
    hedge_selector, sample_selector, imb_selector, _, network_selector = data_filter_selection(feature=False)

    data = process_selection(
        data,
        hedges=hedge_selector,
        samples=sample_selector,
        imbs=imb_selector,
        networks=network_selector
    )

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

    network_dfs = [target_df[target_df['Graph Name'] == g] for g in network_selector]
    fList = target_df['feature'].unique().tolist()

    fig = go.Figure()

    for f, graph_name in zip(network_dfs, network_selector):
        fig.add_trace(
            go.Bar(x=fList, y=f["Avg"], name=graph_name)
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

    col1, col2 = st.columns(2)
    col1.dataframe(target_df[target_df["feature"] == "Mean"])
    col2.dataframe(target_df[target_df["feature"] == "Neighbor"])

    create_parameters_box(
        networks=network_selector,
        imbs=imb_selector,
        sampling=sample_selector,
        hedges=hedge_selector
    )


def plot_hedge_size(data):
    st.title("Mean AUC-PR for hyper edge size")
    _, sample_selection, imb_selection, feature_selection, network_selection = data_filter_selection(hedge=False)

    data = cols_from_setting(data)

    data = process_selection(data, samples=sample_selection, imbs=imb_selection, features=feature_selection)

    data = data[["Graph Name", "hedge"] + list(data.filter(regex="PR"))]
    data = data.groupby(by=["Graph Name", "hedge"], as_index=False).mean()
    data['Avg'] = data[list(data.filter(regex='PR'))].mean(axis=1)
    data.drop(columns=list(data.filter(regex='PR')), inplace=True)

    network_dfs = [data[data['Graph Name'] == g] for g in network_selection]
    fList = data['hedge'].unique().tolist()

    fig = go.Figure()

    for f, graph_name in zip(network_dfs, network_selection):
        fig.add_trace(
            go.Bar(x=fList, y=f["Avg"], name=graph_name)
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

    col1, col2 = st.columns(2)
    col1.dataframe(data[data['hedge'] == 4])
    col2.dataframe(data[data['hedge'] == 5])

    create_parameters_box(
        sampling=sample_selection,
        networks=network_selection,
        features=feature_selection,
        imbs=imb_selection
    )


def plot_imbalance(data):
    st.title("Mean AUC values for different imbalance ratios")
    data = cols_from_setting(data)
    hedge_selection, sample_selection, _, feature_selection, network_selection = data_filter_selection(imb=False)

    data = process_selection(data, samples=sample_selection, hedges=hedge_selection, features=feature_selection)

    data = data[["Graph Name", "imbalance"] + list(data.filter(regex="PR"))]
    data = data.groupby(by=["Graph Name", "imbalance"], as_index=False).mean()
    data['Avg'] = data[list(data.filter(regex="PR"))].mean(axis=1)
    data.drop(columns=list(data.filter(regex="PR")), inplace=True)

    network_dfs = [data[data['Graph Name'] == g] for g in network_selection]
    fList = data['imbalance'].unique().tolist()

    fig = go.Figure()

    for f, graph_name in zip(network_dfs, network_selection):
        fig.add_trace(
            go.Scatter(
                x=fList, y=f['Avg'], name=graph_name,
                line={'width': 7}, marker={'size': 18}
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

    col11, col12 = st.columns(2)
    col11.dataframe(data[data["imbalance"] == 1])
    col12.dataframe(data[data["imbalance"] == 2])
    col21, col22 = st.columns(2)
    col21.dataframe(data[data["imbalance"] == 5])
    col22.dataframe(data[data["imbalance"] == 10])

    create_parameters_box(
        networks=network_selection,
        hedges=hedge_selection,
        sampling=sample_selection,
        features=feature_selection,
    )


def plot_sampling(data):
    st.title("Mean AUC values for different negative sampling techniques")
    data = cols_from_setting(data)
    hedge_selection, _, imb_selection, feature_selection, network_selection = data_filter_selection(sample=False)

    create_parameters_box(
        hedges=hedge_selection,
        imbs=imb_selection,
        features=feature_selection,
        networks=network_selection
    )

    data = process_selection(data, imbs=imb_selection, hedges=hedge_selection, features=feature_selection)
    data = data[["Graph Name", "neg_sample"] + list(data.filter(regex="PR"))]
    data = data.groupby(by=["Graph Name", "neg_sample"], as_index=False).mean()
    data['Avg'] = data[list(data.filter(regex="PR"))].mean(axis=1)
    data = data.drop(columns=list(data.filter(regex='PR')))

    network_dfs = [data[data['Graph Name'] == g] for g in network_selection]
    fList = data['neg_sample'].unique().tolist()

    fig = go.Figure()

    for f, graph_name in zip(network_dfs, network_selection):
        fig.add_trace(
            go.Bar(
                x=fList,
                y=f['Avg'],
                name=graph_name
            )
        )
    """
    cols = st.columns(2)

    cols[0].dataframe(f1)
    cols[1].dataframe(f2)
    """
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

    cols = st.columns(2)

    cols[0].dataframe(data[data.neg_sample == "clique"])
    cols[1].dataframe(data[data.neg_sample == "star"])
    st.dataframe(data[data.neg_sample == "tailed"])
