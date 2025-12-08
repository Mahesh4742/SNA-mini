import os
from typing import Dict, List

import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

from sna_analysis import (
    DATA_DIR,
    basic_network_stats,
    build_traffic_graph,
    compute_centrality_measures,
    load_domestic_data,
    load_international_data,
)

ZONE_CITY_SETS: Dict[str, List[str]] = {
    "North": [
        "DELHI",
        "AMRITSAR",
        "CHANDIGARH",
        "JAIPUR",
        "LUCKNOW",
        "VARANASI",
        "DEHRADUN",
        "SRINAGAR",
        "JAMMU",
        "LEH",
        "AGRA",
        "KANPUR",
        "PATNA",
        "RAIPUR",
        "RANCHI",
        "BHOPAL",
    ],
    "South": [
        "BENGALURU",
        "CHENNAI",
        "HYDERABAD",
        "KOCHI",
        "THIRUVANANTHAPURAM",
        "KOZHIKODE",
        "COIMBATORE",
        "MADURAI",
        "MANGALURU",
        "VISAKHAPATNAM",
        "TIRUCHIRAPPALLI",
        "VIJAYAWADA",
        "TRIVANDRUM",
    ],
    "Central": [
        "MUMBAI",
        "PUNE",
        "AHMEDABAD",
        "SURAT",
        "NAGPUR",
        "INDORE",
        "BHUBANESWAR",
        "GOA",
        "AURANGABAD",
        "VADODARA",
        "UDAIPUR",
        "JODHPUR",
    ],
}


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


def filter_domestic_by_zone(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    if zone == "All":
        return df

    zone_set = {city.upper() for city in ZONE_CITY_SETS.get(zone, [])}
    if not zone_set:
        return df

    city1 = df["CITY 1"].astype(str).str.upper()
    city2 = df["CITY 2"].astype(str).str.upper()
    mask = city1.isin(zone_set) & city2.isin(zone_set)
    return df.loc[mask].reset_index(drop=True)


def build_centrality_chart(
    data: pd.DataFrame,
    metric: str,
    color: str,
    title: str,
) -> alt.Chart:
    chart_df = data.reset_index().rename(columns={"index": "City"})
    chart_df["City"] = chart_df["City"].astype(str).str.title()
    return (
        alt.Chart(chart_df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(metric, title=metric),
            y=alt.Y("City", sort="-x"),
            tooltip=["City", alt.Tooltip(metric, format=".4f")],
        )
        .properties(title=title)
    )


def build_route_chart(df: pd.DataFrame, limit: int = 15) -> alt.Chart | None:
    if not {"PASSENGERS TO CITY 2", "PASSENGERS FROM CITY 2"}.issubset(df.columns):
        return None

    temp = df.copy()
    temp["TOTAL_PASSENGERS"] = temp["PASSENGERS TO CITY 2"].fillna(0) + temp[
        "PASSENGERS FROM CITY 2"
    ].fillna(0)
    temp = (
        temp.sort_values("TOTAL_PASSENGERS", ascending=False)
        .head(limit)
        .assign(
            Route=lambda d: d["CITY 1"].astype(str).str.title()
            + " → "
            + d["CITY 2"].astype(str).str.title()
        )
    )

    if temp.empty:
        return None

    return (
        alt.Chart(temp)
        .mark_bar(color="#f39c12")
        .encode(
            x=alt.X("TOTAL_PASSENGERS", title="Total Passengers"),
            y=alt.Y("Route", sort="-x", title="City Pair"),
            tooltip=["Route", alt.Tooltip("TOTAL_PASSENGERS", format=",.0f")],
        )
        .properties(title="Top Routes by Passenger Volume")
    )


def build_network_graph_figure(
    G: nx.Graph,
    centrality_df: pd.DataFrame,
    title: str,
    max_edges: int,
) -> plt.Figure | None:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return None

    working_graph = G.copy()
    if max_edges and working_graph.number_of_edges() > max_edges:
        sorted_edges = sorted(
            working_graph.edges(data=True),
            key=lambda edge: edge[2].get("weight", 1.0),
            reverse=True,
        )
        working_graph = nx.Graph()
        for u, v, data in sorted_edges[:max_edges]:
            working_graph.add_edge(u, v, **data)

    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))
    if working_graph.number_of_nodes() == 0:
        return None

    centrality_subset = centrality_df.reindex(working_graph.nodes()).fillna(0)
    degree_map = centrality_subset["Degree Centrality"].to_dict()
    betweenness_map = centrality_subset["Betweenness Centrality"].to_dict()

    if working_graph.number_of_nodes() <= 80:
        pos = nx.kamada_kawai_layout(working_graph, weight="weight")
    else:
        pos = nx.spring_layout(
            working_graph, weight="weight", k=0.3, iterations=200, seed=42
        )

    node_sizes = [
        max(betweenness_map.get(node, 0.0), 1e-5) * 60000 for node in working_graph.nodes()
    ]
    node_colors = [degree_map.get(node, 0.0) for node in working_graph.nodes()]

    fig, ax = plt.subplots(figsize=(10, 8))
    nodes = nx.draw_networkx_nodes(
        working_graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        alpha=0.85,
        ax=ax,
    )
    nx.draw_networkx_edges(
        working_graph,
        pos,
        width=_compute_edge_widths(working_graph),
        alpha=0.18,
        edge_color="#7f8c8d",
        ax=ax,
    )

    top_labels = (
        centrality_subset.sort_values("Betweenness Centrality", ascending=False)
        .head(10)
        .index.tolist()
    )
    labels = {node: node.title() for node in top_labels if node in working_graph.nodes()}
    nx.draw_networkx_labels(
        working_graph, pos, labels=labels, font_size=9, font_weight="bold", ax=ax
    )

    fig.colorbar(nodes, ax=ax, label="Degree Centrality")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


def build_interactive_network_html(
    G: nx.Graph,
    centrality_df: pd.DataFrame,
    title: str,
    max_edges: int,
) -> str | None:
    """Build an interactive PyVis network (zoom + drag)."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return None

    working_graph = G.copy()
    if max_edges and working_graph.number_of_edges() > max_edges:
        sorted_edges = sorted(
            working_graph.edges(data=True),
            key=lambda edge: edge[2].get("weight", 1.0),
            reverse=True,
        )
        working_graph = nx.Graph()
        for u, v, data in sorted_edges[:max_edges]:
            working_graph.add_edge(u, v, **data)

    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))
    if working_graph.number_of_nodes() == 0:
        return None

    centrality_subset = centrality_df.reindex(working_graph.nodes()).fillna(0)
    degree_map = centrality_subset["Degree Centrality"].to_dict()
    betweenness_map = centrality_subset["Betweenness Centrality"].to_dict()

    if working_graph.number_of_nodes() <= 80:
        pos = nx.kamada_kawai_layout(working_graph, weight="weight")
    else:
        pos = nx.spring_layout(
            working_graph, weight="weight", k=0.3, iterations=200, seed=42
        )

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50",
        directed=False,
        notebook=False,
        cdn_resources="remote",
    )
    net.barnes_hut()

    for node in working_graph.nodes():
        net.add_node(
            node,
            label=node.title(),
            value=degree_map.get(node, 0.0),
            size=max(betweenness_map.get(node, 0.0), 1e-5) * 120,
            x=float(pos[node][0]) * 1000,
            y=float(pos[node][1]) * 1000,
            physics=True,
            color=None,
        )

    for u, v, data in working_graph.edges(data=True):
        weight = data.get("weight", 1.0)
        net.add_edge(u, v, value=weight, width=0.5 + weight * 2)

    net.set_options(
        """
        {
          "interaction": { "dragNodes": true, "zoomView": true, "dragView": true },
          "physics": {
            "stabilization": true,
            "barnesHut": { "springLength": 90, "damping": 0.2 }
          },
          "nodes": { "shape": "dot", "scaling": { "min": 3, "max": 60 } },
          "edges": { "color": "#7f8c8d", "smooth": false }
        }
        """
    )

    # Heading is already shown in Streamlit; avoid double titles in the embed
    net.heading = ""
    return net.generate_html(notebook=False)


def _compute_edge_widths(graph: nx.Graph) -> List[float]:
    weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]
    if not weights:
        return [0.8] * graph.number_of_edges()
    max_weight = max(weights)
    if max_weight == 0:
        return [0.8] * graph.number_of_edges()
    return [0.4 + (w / max_weight) * 4.0 for w in weights]


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

    selected_zone = "All"
    if network_type == "Domestic":
        zone_options = ["All", *ZONE_CITY_SETS.keys()]
        selected_zone = st.sidebar.selectbox(
            "Filter domestic network by zone",
            options=zone_options,
            index=0,
            help="Keeps only the routes where both cities belong to the selected zone.",
        )

    max_edges_display = st.sidebar.slider(
        "Max edges to display in network graph",
        min_value=50,
        max_value=1000,
        value=250 if network_type == "Domestic" else 200,
        step=50,
    )

    with st.spinner(f"Loading {network_type.lower()} data and building network..."):
        df = load_data(network_type).copy()

        if network_type == "Domestic":
            df = filter_domestic_by_zone(df, selected_zone)

        if df.empty:
            st.warning(
                "No routes available for the selected filters. Try a different configuration."
            )
            return

        G, centrality_df, stats = build_graph_and_metrics(df)

    st.subheader(f"{network_type} Network Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Cities (Nodes)", f"{stats['num_nodes']}")
    col2.metric("Number of Routes (Edges)", f"{stats['num_edges']}")
    col3.metric("Network Density", f"{stats['density']:.4f}")

    if network_type == "Domestic":
        st.caption(
            f"Showing results for **{selected_zone}** zone"
            if selected_zone != "All"
            else "Showing results for all zones"
        )

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
            width="stretch",
        )

    with tab2:
        st.dataframe(
            betweenness_top.style.background_gradient(
                cmap="Greens", subset=["Betweenness Centrality"]
            ),
            width="stretch",
        )

    with tab3:
        st.dataframe(
            closeness_top.style.background_gradient(
                cmap="Oranges", subset=["Closeness Centrality"]
            ),
            width="stretch",
        )

    st.subheader("Centrality Bar Charts")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        degree_chart = build_centrality_chart(
            degree_top,
            "Degree Centrality",
            "#1f77b4",
            "Top Degree Centrality",
        )
        st.altair_chart(degree_chart, width="stretch")

    with chart_col2:
        betweenness_chart = build_centrality_chart(
            betweenness_top,
            "Betweenness Centrality",
            "#2ca02c",
            "Top Betweenness Centrality",
        )
        st.altair_chart(betweenness_chart, width="stretch")

    st.markdown("---")

    if network_type == "Domestic":
        st.subheader("Route Traffic – Selected Zone")
        route_chart = build_route_chart(df)
        if route_chart is not None:
            st.altair_chart(route_chart, width="stretch")
        else:
            st.info("Passenger columns not available to build route chart.")

    st.subheader(f"Sample Routes Table – {network_type}")
    st.caption(
        "Showing the city-pair data used to construct the network."
    )
    st.dataframe(df.head(200), width="stretch")

    st.markdown("---")
    st.subheader("Network Graph")
    graph_title = (
        f"{network_type} Network Graph – {selected_zone} Zone"
        if network_type == "Domestic" and selected_zone != "All"
        else f"{network_type} Network Graph"
    )
    interactive_html = build_interactive_network_html(
        G, centrality_df, graph_title, max_edges_display
    )
    if interactive_html is not None:
        components.html(interactive_html, height=780, scrolling=True)
    else:
        st.info("Unable to generate network graph for the current selection.")



if __name__ == "__main__":
    main()


