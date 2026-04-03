Supply Chain Risk Modelling: Quantitative country & corridor risk scoring using WorldRiskIndex data, engineered features, and ML/DL forecasting

Overview
This project builds a quantitative framework for assessing supply chain corridor risk at the country level. The core question it tries to answer is - how reliable a route is, as a logistics node, and how catastrophic its unavailability would be for the broader trade network.
The pipeline runs across three notebooks: data preparation, feature engineering & modelling, and visualization. It produces a composite country risk score that integrates five engineered risk dimensions with a strategic supply chain concentration layer, validated through both classical ML classifiers and an LSTM forecasting model.
Data Sources
WorldRiskIndex (WRI) — Primary Dataset
Published annually by Bündnis Entwicklung Hilft and Ruhr University Bochum. The dataset covers 193 countries across 26 years (1998–2023) and decomposes national disaster risk into two top-level dimensions:

•	Exposure (E) — natural hazard intensity across seven domains: earthquakes, tsunamis, coastal flooding, riverine flooding, cyclones, droughts, and sea level rise.
•	Vulnerability (V) — the composite of Susceptibility (S), Lack of Coping Capacity (C), and Lack of Adaptive Capacity (A), covering governance, infrastructure, economic fragility, and social resilience.
Each country-year row contains raw base values (_Base columns), 0–100 normalized scores (_Norm columns), and all intermediate aggregates — roughly 248 columns in total. The dataset is available at worldriskindex.org.
I’ve downloaded the dataset from: The Humanitarian Data Exchange (https://data.humdata.org)
SCCR — Manual Scoring Table
The Supply Chain Concentration Risk scores were built manually for 21 strategic nodes, drawing on UNCTAD maritime statistics (container throughput data), Bimco chokepoint reports, and Lloyd's List port traffic rankings. This layer was necessary because no public dataset directly quantifies how substitutable a routing node is — a gap that becomes critical when modelling corridors like Malacca or Suez.
Project Structure
Notebook 1 — SCRM_Data_Prep.ipynb
Handles everything upstream of feature engineering. The notebook loads the raw WRI dataset, maps its three-level column hierarchy (composites → components → domain indicators → normalized leaf indicators), cleans naming conventions, runs completeness checks including sentinel value analysis, and exports three structured tables: a full time-series panel, a latest-year cross-sectional snapshot with global ranks, and a country-level summary with historical statistics.

Notebook 2 — SCRM_Feature_Eng_Model.ipynb
The analytical core of the project. Six features are engineered from the WRI sub-indicators, each designed to capture a distinct dimension of supply chain risk:

•	Natural Hazard Exposure Score (NHES) — seven WRI hazard domains reweighted by logistics disruption severity rather than humanitarian impact.
•	Governance and Stability Score (GSS) — WRI state and government indicators normalized and inverted so higher always means more governance risk.
•	Infrastructure Resilience Score (IRS) — connectivity and services indicators weighted toward electricity and communications capacity.
•	Economic Fragility Score (EFS) — income level, aid dependency, and price instability, reflecting fiscal capacity to restore logistics operations.
•	Recent Shock Burden Score (RSBS) — five-year rolling disaster and conflict burden, with conflict weighted higher for its total logistics shutdown effect.
•	Supply Chain Concentration Risk (SCCR) — a manual scoring layer capturing strategic node criticality (throughput volume, chokepoint control, substitutability). Applied as a post-composite impact multiplier rather than a weighted feature, keeping probability and consequence analytically separated.

These features combine into a three-stage composite: a base CCRS (five-feature weighted composite), a compound-adjusted CCRS that applies non-linear amplification when multiple dimensions elevate simultaneously, and a final CCRS that incorporates the SCCR impact multiplier.

Route-level risk is computed across four aggregation methods (simple weighted average, bottleneck, chokepoint-adjusted, and probabilistic cascade) using Vietnam–Rotterdam as the reference corridor. Two ML classifiers — Random Forest and Gradient Boosting — validate the hand-crafted feature weights through feature importance comparisons. An LSTM network provides sequence-aware next-year risk forecasting using five-year lookback windows.
Notebook 3 — SCRM_Data_Viz.ipynb
Fourteen charts covering the full analytical narrative: feature distributions, correlation structure, country risk rankings with dimension decomposition, temporal trajectories with crisis annotations, route comparisons across all four aggregation methods, chokepoint radar profiles, drawdown analysis, volatility-versus-level quadrant analysis, a multi-decade risk heatmap, and a side-by-side comparison of manual feature weights versus ML-derived importance scores.

Technical Stack
•	Languages & Core Libraries Python 3.10+ with pandas, NumPy, scikit-learn, TensorFlow/Keras
•	Machine Learning & Deep Learning Random Forest and Gradient Boosting classifiers for risk tier classification; LSTM network for one-year-ahead CCRS forecasting
•	Validation Temporal train/test splits (pre-2022 train, 2022+ test) to prevent data leakage in all models
•	Visualization Matplotlib, Seaborn, Plotly etc. for the full visualization suite
