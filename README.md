# Indian Air Traffic Social Network Analysis (SNA)

A comprehensive Social Network Analysis (SNA) dashboard for analyzing Indian air traffic patterns, both domestic and international. This project uses NetworkX to build and analyze air traffic networks, providing insights into city connectivity, centrality measures, and route traffic patterns.

## 📋 Overview

This project analyzes air traffic data to understand the network structure of Indian aviation. It processes city-pair data from domestic and international flights, constructs network graphs, and computes various centrality measures to identify key hubs and connections in the air traffic network.

## ✨ Features

- **Interactive Streamlit Dashboard**: User-friendly web interface for exploring air traffic networks
- **Dual Network Analysis**: Separate analysis for domestic and international air traffic
- **Zone-based Filtering**: Filter domestic networks by geographic zones (North, South, Central)
- **Centrality Measures**: 
  - Degree Centrality
  - Betweenness Centrality
  - Closeness Centrality
- **Visualizations**:
  - Interactive network graphs with node sizing and coloring based on centrality
  - Bar charts for top cities by centrality measures
  - Route traffic analysis charts
  - Network statistics and metrics
- **Data Processing**: Automated loading and processing of Excel files containing city-pair data

## 🚀 Installation

### Prerequisites

- Python 3.13+ (or Python 3.11+)
- pip package manager

### Setup Steps

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   The requirements include:
   - `streamlit` - Web dashboard framework
   - `networkx` - Network analysis library
   - `pandas` - Data manipulation
   - `numpy` - Numerical computing
   - `matplotlib` - Network graph visualization
   - `altair` - Interactive charts
   - `openpyxl` - Excel file reading
   - `pyarrow` - Data format support

3. **Ensure data files are in place**:
   - Place your Excel data files in the `Data/` directory
   - Required files:
     - Domestic: `DOM CITYPAIR DATA, [MONTH] 2025.xlsx` files
     - International: `25Q1_4.xlsx`, `25Q2_4.xlsx`

## 📖 Usage

### Running the Streamlit Dashboard

1. **Start the dashboard**:
   ```bash
   streamlit run sna_dashboard_streamlit.py
   ```

2. **Access the dashboard**:
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

3. **Using the Dashboard**:
   - Select network type (Domestic or International) from the sidebar
   - For domestic networks, choose a zone filter (All, North, South, Central)
   - Adjust the maximum number of edges to display in the network graph
   - Explore the various tabs and visualizations

### Using the Analysis Module

You can also use the `sna_analysis.py` module programmatically:

```python
from sna_analysis import (
    load_domestic_data,
    load_international_data,
    build_traffic_graph,
    compute_centrality_measures,
    basic_network_stats
)

# Load data
domestic_df = load_domestic_data()

# Build network graph
G = build_traffic_graph(domestic_df)

# Compute centrality measures
centrality_df = compute_centrality_measures(G)

# Get network statistics
stats = basic_network_stats(G)
```

## 📁 Project Structure

```
SNA-mini-main/
│
├── Data/                          # Data directory
│   ├── DOM CITYPAIR DATA, *.xlsx  # Domestic flight data files
│   ├── 25Q1_4.xlsx               # International flight data (Q1)
│   └── 25Q2_4.xlsx               # International flight data (Q2)
│
├── sna_analysis.py               # Core analysis functions
├── sna_dashboard_streamlit.py    # Streamlit dashboard application
├── SNA.ipynb                     # Jupyter notebook (if available)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔍 Key Components

### `sna_analysis.py`
Core analysis module containing:
- `load_domestic_data()` - Loads and combines domestic flight data
- `load_international_data()` - Loads and combines international flight data
- `build_traffic_graph()` - Constructs NetworkX graph from city-pair data
- `compute_centrality_measures()` - Calculates centrality metrics
- `basic_network_stats()` - Returns network summary statistics

### `sna_dashboard_streamlit.py`
Streamlit web application providing:
- Interactive network type selection
- Zone-based filtering for domestic routes
- Centrality measure visualizations
- Network graph visualization
- Route traffic analysis
- Data tables and statistics

## 📊 Data Format

The application expects Excel files with the following structure:
- **Header row**: Column names in row 3 (0-based index 2)
- **Required columns**:
  - `CITY 1` - Origin city
  - `CITY 2` - Destination city
  - `PASSENGERS TO CITY 2` - Passengers traveling to city 2
  - `PASSENGERS FROM CITY 2` - Passengers traveling from city 2
- **Optional columns**: Freight and mail data

## 🎯 Network Analysis Metrics

### Degree Centrality
Measures the number of direct connections a city has. Higher values indicate more direct routes.

### Betweenness Centrality
Measures how often a city appears on shortest paths between other cities. Identifies important transit hubs.

### Closeness Centrality
Measures how close a city is to all other cities in the network. Indicates accessibility.

## 🛠️ Technologies Used

- **Python 3.13+**
- **Streamlit** - Web application framework
- **NetworkX** - Network analysis and graph algorithms
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Network graph visualization
- **Altair** - Interactive statistical visualizations
- **NumPy** - Numerical computing

## 📝 Notes

- The dashboard uses caching to improve performance when switching between views
- Network graphs are automatically simplified when they exceed the maximum edge limit
- The application supports both weighted and unweighted network analysis
- Geographic zones (North, South, Central) are predefined for Indian cities

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Note**: Make sure all required data files are present in the `Data/` directory before running the application. Missing files will result in errors.

