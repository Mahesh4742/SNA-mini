import os
from typing import List, Tuple

import networkx as nx
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")


DOMESTIC_FILES: List[str] = [
    "DOM CITYPAIR DATA, JANUARY 2025.xlsx",
    "DOM CITYPAIR DATA, FEBRUARY 2025.xlsx",
    "DOM CITYPAIR DATA, MARCH 2025.xlsx",
    "DOM CITYPAIR DATA, APRIL 2025.xlsx",
    "DOM CITYPAIR DATA, MAY 2025.xlsx",
    "DOM CITYPAIR DATA, JUNE 2025.xlsx",
    "DOM CITYPAIR DATA, JULY 2025.xlsx",
    "DOM CITYPAIR DATA, AUGUST 2025.xlsx",
]

INTERNATIONAL_FILES: List[str] = [
    "25Q1_4.xlsx",
    "25Q2_4.xlsx",
]


def _load_citypair_file(path: str) -> pd.DataFrame:
    """
    Load a city-pair Excel file that has a multi-row header similar to the
    DOM CITYPAIR DATA files used in the notebook.

    Assumes:
      - Row 2 (0-based index) contains the column names
      - Data starts from row 3 onwards
    """
    raw_df = pd.read_excel(path, header=None)

    # Extract and clean column names from the third row (index 2)
    column_names = (
        raw_df.iloc[2]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

    df = raw_df[3:].copy()
    df.columns = column_names.tolist()
    df = df.reset_index(drop=True)

    # Drop columns whose names are literally 'nan'
    df = df.loc[:, df.columns != "nan"]

    # Standardize column names: collapse multiple spaces and strip
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

    # Ensure expected numeric columns exist (some international files may differ;
    # we only convert the ones that are present).
    numeric_cols = [
        "PASSENGERS TO CITY 2",
        "PASSENGERS FROM CITY 2",
        "FREIGHT TO CITY 2",
        "FREIGHT FROM CITY 2",
        "MAIL TO CITY 2",
        "MAIL FROM CITY 2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where key city identifiers are missing
    for col in ["CITY 1", "CITY 2"]:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Drop 'S.No.' if present – it's just a row index
    if "S.No." in df.columns:
        df = df.drop(columns=["S.No."])

    return df


def load_domestic_data(
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Load and vertically concatenate all domestic flight city-pair datasets."""
    base_dir = data_dir or DATA_DIR
    frames: List[Tuple[str, pd.DataFrame]] = []

    for fname in DOMESTIC_FILES:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            continue
        df = _load_citypair_file(path)
        df["SOURCE_FILE"] = fname
        frames.append((fname, df))

    if not frames:
        raise FileNotFoundError(
            f"No domestic files found in {base_dir}. "
            f"Expected one or more of: {', '.join(DOMESTIC_FILES)}"
        )

    combined = pd.concat([df for _, df in frames], ignore_index=True)
    return combined


def load_international_data(
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Load and vertically concatenate all international flight datasets."""
    base_dir = data_dir or DATA_DIR
    frames: List[Tuple[str, pd.DataFrame]] = []

    for fname in INTERNATIONAL_FILES:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            continue
        df = _load_citypair_file(path)
        df["SOURCE_FILE"] = fname
        frames.append((fname, df))

    if not frames:
        raise FileNotFoundError(
            f"No international files found in {base_dir}. "
            f"Expected one or more of: {', '.join(INTERNATIONAL_FILES)}"
        )

    combined = pd.concat([df for _, df in frames], ignore_index=True)
    return combined


def build_traffic_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Construct an undirected city graph from a city-pair traffic DataFrame.

    Expects at least:
      - 'CITY 1'
      - 'CITY 2'
      - 'PASSENGERS TO CITY 2'
      - 'PASSENGERS FROM CITY 2'
    """
    if not {"CITY 1", "CITY 2"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'CITY 1' and 'CITY 2' columns.")

    # Compute total passengers if the columns are available
    to_col = "PASSENGERS TO CITY 2"
    from_col = "PASSENGERS FROM CITY 2"

    total_passengers = None
    if to_col in df.columns and from_col in df.columns:
        total_passengers = df[to_col].fillna(0) + df[from_col].fillna(0)
    elif to_col in df.columns:
        total_passengers = df[to_col].fillna(0)
    elif from_col in df.columns:
        total_passengers = df[from_col].fillna(0)

    G = nx.Graph()

    for idx, row in df.iterrows():
        city1 = row["CITY 1"]
        city2 = row["CITY 2"]
        if pd.isna(city1) or pd.isna(city2):
            continue

        weight = float(total_passengers.iloc[idx]) if total_passengers is not None else 1.0
        G.add_edge(str(city1), str(city2), weight=weight)

    return G


def compute_centrality_measures(G: nx.Graph) -> pd.DataFrame:
    """
    Compute degree, betweenness, and closeness centrality for a graph.
    Uses the 'weight' edge attribute for betweenness/closeness where present.
    """
    degree_c = nx.degree_centrality(G)

    # Use 'weight' as edge weight if it exists, otherwise unweighted
    weight_key = "weight" if any("weight" in data for *_ , data in G.edges(data=True)) else None

    betweenness_c = nx.betweenness_centrality(G, weight=weight_key)
    closeness_c = nx.closeness_centrality(G, distance=weight_key)

    df = pd.DataFrame(
        {
            "Degree Centrality": pd.Series(degree_c),
            "Betweenness Centrality": pd.Series(betweenness_c),
            "Closeness Centrality": pd.Series(closeness_c),
        }
    )

    return df


def basic_network_stats(G: nx.Graph) -> dict:
    """Return simple summary statistics for a graph."""
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
    }


def run_full_analysis(data_dir: str | None = None) -> dict:
    """
    Convenience function to run the full pipeline for both domestic and
    international networks. Returns a dictionary of results that can be
    consumed by a dashboard or notebook.
    """
    base_dir = data_dir or DATA_DIR

    domestic_df = load_domestic_data(base_dir)
    international_df = load_international_data(base_dir)

    domestic_G = build_traffic_graph(domestic_df)
    international_G = build_traffic_graph(international_df)

    domestic_centrality = compute_centrality_measures(domestic_G)
    international_centrality = compute_centrality_measures(international_G)

    return {
        "domestic": {
            "df": domestic_df,
            "graph": domestic_G,
            "centrality": domestic_centrality,
            "stats": basic_network_stats(domestic_G),
        },
        "international": {
            "df": international_df,
            "graph": international_G,
            "centrality": international_centrality,
            "stats": basic_network_stats(international_G),
        },
    }


