import os

import pandas as pd
import streamlit as st

from sna_analysis import (
    DATA_DIR,
    basic_network_stats,
    build_traffic_graph,
    compute_centrality_measures,
    load_domestic_data,
    load_international_data,
)


st.set_page_config(
    page_title="Air Traffic Social Network Analysis",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(kind: str) -> pd.DataFrame:
    if kind == "Domestic":
        return load_domestic_data(DATA_DIR)
    else:
        return load_international_data(DATA_DIR)


@st.cache_data(show_spinner=False)
def build_graph_and_metrics(df: pd.DataFrame):
    G = build_traffic_graph(df)
    centrality_df = compute_centrality_measures(G)
    stats = basic_network_stats(G)
    return G, centrality_df, stats


def main() -> None:
    st.title("Indian Air Traffic Social Network Analysis")

    st.sidebar.header("Configuration")
    network_type = st.sidebar.radio(
        "Select network type",
        options=["Domestic", "International"],
        index=0,
    )

    st.sidebar.markdown(
        f"**Data folder:** `{os.path.abspath(DATA_DIR)}`"
    )

    with st.spinner(f"Loading {network_type.lower()} data and building network..."):
        df = load_data(network_type)
        G, centrality_df, stats = build_graph_and_metrics(df)

    st.subheader(f"{network_type} Network Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Cities (Nodes)", f"{stats['num_nodes']}")
    col2.metric("Number of Routes (Edges)", f"{stats['num_edges']}")
    col3.metric("Network Density", f"{stats['density']:.4f}")

    st.markdown("---")

    st.subheader(f"Top Cities by Centrality – {network_type}")

    # Sort once and reuse for different views
    degree_top = centrality_df.sort_values(
        by="Degree Centrality", ascending=False
    ).head(15)
    betweenness_top = centrality_df.sort_values(
        by="Betweenness Centrality", ascending=False
    ).head(15)
    closeness_top = centrality_df.sort_values(
        by="Closeness Centrality", ascending=False
    ).head(15)

    tab1, tab2, tab3 = st.tabs(
        ["Degree Centrality", "Betweenness Centrality", "Closeness Centrality"]
    )

    with tab1:
        st.dataframe(
            degree_top.style.background_gradient(
                cmap="Blues", subset=["Degree Centrality"]
            ),
            use_container_width=True,
        )

    with tab2:
        st.dataframe(
            betweenness_top.style.background_gradient(
                cmap="Greens", subset=["Betweenness Centrality"]
            ),
            use_container_width=True,
        )

    with tab3:
        st.dataframe(
            closeness_top.style.background_gradient(
                cmap="Oranges", subset=["Closeness Centrality"]
            ),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader(f"Sample Routes Table – {network_type}")
    st.caption(
        "Showing a the city-pair data used to construct the network."
    )
    st.dataframe(df, use_container_width=True)



if __name__ == "__main__":
    main()


