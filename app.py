# =============================================================================
# app.py — SCRM Supply Chain Risk Intelligence Dashboard
# Deploy to: Streamlit Community Cloud (share.streamlit.io)
#
# Four pages mirroring your Power BI dashboard structure:
#   1. Global Risk Intelligence Map
#   2. Route Builder & Comparison
#   3. Country Deep Dive
#   4. Portfolio Risk Summary
#
# LSTM NOTE: Forecasts are loaded from lstm_forecasts.csv at repo root (pre-computed).
# This keeps TensorFlow out of the deployed environment, saving ~500 MB RAM.
# The notebook exports these exact columns: iso3, country, year,
# ccrs_forecast, ccrs_forecast_hi, ccrs_forecast_lo. The app works fine without it.
# =============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config must be the very first Streamlit call in the script ──────────
st.set_page_config(
    page_title="SCRM — Supply Chain Risk Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Risk tier colour map — red = high risk, applied consistently across all pages.
# Blue → Green → Orange → Red mirrors the severity progression.
TIER_COLORS = {
    'Low':         '#2196F3',   # Blue
    'Medium-Low':  '#4CAF50',   # Green
    'Medium-High': '#FF9800',   # Amber
    'High':        '#F44336',   # Red
}

# Risk regime colours used in pie charts and badges
REGIME_COLORS = {
    'improving':        '#4CAF50',
    'stable':           '#FF9800',
    'deteriorating':    '#F44336',
    'insufficient_data':'#9E9E9E',
}

# Human-readable labels for the five probability-layer dimensions
DIM_LABELS = {
    'nhes':     'Hazard Exposure (NHES)',
    'gss_risk': 'Governance Risk (GSS)',
    'irs':      'Infra. Weakness (IRS)',
    'efs':      'Economic Fragility (EFS)',
    'rsbs':     'Shock Burden (RSBS)',
}

# Human-readable labels for route scoring methods
METHOD_LABELS = {
    'simple_avg':    'Simple Weighted Avg',
    'bottleneck':    'Bottleneck (λ=0.4)',
    'chokepoint':    'Chokepoint Multiplier',
    'probabilistic': 'Probabilistic Cascade',
}

# =============================================================================
# DATA LOADING — @st.cache_data means these functions only run once per session.
# Without caching, Streamlit re-runs the entire script on every user interaction,
# meaning your CSVs would reload every time a slider moves. Caching prevents that.
# =============================================================================

@st.cache_data
def load_features():
    """Load the main country risk features table produced by SCRM_Feature_Eng_Model."""
    path = 'country_risk_features.csv'
    if not os.path.exists(path):
        st.error(f"Missing file: {path}. Make sure this CSV is in your repository root.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_routes():
    """Load the pre-computed route risk scores table."""
    path = 'route_risk_scores.csv'
    if not os.path.exists(path):
        st.error(f"Missing file: {path}. Make sure this CSV is in your repository root.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_forecasts():
    """
    Load pre-computed LSTM forecasts if available.

    The notebook exports these exact columns:
        iso3, country, year, ccrs_forecast, ccrs_forecast_hi, ccrs_forecast_lo, is_forecast

    If the file doesn't exist, the app simply hides the forecast overlay.
    """
    path = 'lstm_forecasts.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# Load all data at module level so every page can access it
df           = load_features()
routes_df    = load_routes()
forecasts_df = load_forecasts()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🌍 SCRM Dashboard")
st.sidebar.markdown(
    "**Supply Chain Risk Intelligence**  \n"
    "WorldRiskIndex · 193 Countries · 26 Years"
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    options=[
        "📍 Global Risk Map",
        "🛳️ Route Builder",
        "🔍 Country Deep Dive",
        "📊 Portfolio Summary",
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Data: WorldRiskIndex 2000–2023  \n"
    "Models: RF + GB Classifiers · LSTM Forecaster  \n"
    "Scoring: Simple Avg · Bottleneck · Chokepoint · Probabilistic"
)

# =============================================================================
# PAGE 1 — GLOBAL RISK INTELLIGENCE MAP
# Choropleth of ccrs_final with year slider, KPI cards, and tier distribution.
# =============================================================================

if page == "📍 Global Risk Map":

    st.title("Global Risk Intelligence Map")
    st.markdown(
        "Country-level composite risk scores (CCRS Final) integrating natural hazard exposure, "
        "governance fragility, infrastructure resilience, economic stress, and strategic "
        "concentration risk (SCCR). Use the year slider to track how the global risk landscape "
        "has shifted since 2000."
    )

    # Year selector — placed above the map so it feels like a timeline control
    years = sorted(df['year'].unique())
    selected_year = st.slider(
        "Year",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=int(max(years))
    )

    df_year = df[df['year'] == selected_year].copy()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label="Avg Global Risk Score",
        value=f"{df_year['ccrs_final'].mean():.1f}",
        help="Mean CCRS Final across all 193 countries for the selected year"
    )
    c2.metric(
        label="High Risk Countries",
        value=int((df_year['risk_tier'] == 'High').sum()),
        help="Countries with CCRS Final ≥ 60"
    )
    c3.metric(
        label="Deteriorating Regimes",
        value=int((df_year['risk_regime'] == 'deteriorating').sum()),
        help="Countries with a positive 5-year CCRS trend slope > 0.3"
    )
    c4.metric(
        label="Improving Regimes",
        value=int((df_year['risk_regime'] == 'improving').sum()),
        help="Countries with a negative 5-year CCRS trend slope < -0.3"
    )

    # ── Choropleth map ────────────────────────────────────────────────────────
    # RdYlBu_r → reversed so blue = low risk, red = high risk.
    # hover_data exposes the most analytically relevant fields for recruiters
    # reviewing the dashboard — risk_tier, sccr, and regime in one hover.
    fig_map = px.choropleth(
        df_year,
        locations='iso3',
        color='ccrs_final',
        hover_name='country',
        hover_data={
            'ccrs_final':   ':.1f',
            'risk_tier':    True,
            'risk_regime':  True,
            'sccr':         ':.1f',
            'ccrs_drawdown':':.1f',
            'iso3':         False,
        },
        color_continuous_scale='RdYlBu_r',
        range_color=(0, 100),
        labels={'ccrs_final': 'CCRS Final Score'},
        title=f"CCRS Final Score — {selected_year}"
    )
    fig_map.update_layout(
        height=520,
        coloraxis_colorbar=dict(title="Risk Score<br>(0–100)"),
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(showframe=False, showcoastlines=True)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Risk tier bar chart ───────────────────────────────────────────────────
    st.subheader("Risk Tier Distribution")
    tier_order  = ['Low', 'Medium-Low', 'Medium-High', 'High']
    tier_counts = (
        df_year['risk_tier']
        .value_counts()
        .reindex(tier_order)
        .fillna(0)
        .reset_index()
    )
    tier_counts.columns = ['Tier', 'Count']

    fig_tier = px.bar(
        tier_counts,
        x='Tier',
        y='Count',
        color='Tier',
        color_discrete_map=TIER_COLORS,
        text='Count',
        labels={'Count': 'Number of Countries'},
    )
    fig_tier.update_traces(textposition='outside')
    fig_tier.update_layout(showlegend=False, height=280, xaxis_title=None)
    st.plotly_chart(fig_tier, use_container_width=True)

    # ── Top 10 riskiest countries ─────────────────────────────────────────────
    st.subheader(f"Top 10 Highest Risk Countries — {selected_year}")
    top10 = (
        df_year
        .nlargest(10, 'ccrs_final')
        [['country', 'ccrs_final', 'risk_tier', 'risk_regime', 'sccr', 'ccrs_drawdown', 'ccrs_vol_5y']]
        .reset_index(drop=True)
    )
    top10.index += 1
    top10.columns = ['Country', 'CCRS Final', 'Tier', 'Regime', 'SCCR', 'Drawdown', '5Y Vol']
    st.dataframe(
        top10.style.format({
            'CCRS Final': '{:.1f}',
            'SCCR':       '{:.1f}',
            'Drawdown':   '{:.1f}',
            '5Y Vol':     '{:.2f}',
        }),
        use_container_width=True
    )


# =============================================================================
# PAGE 2 — ROUTE BUILDER & COMPARISON
# Compare three Vietnam → Rotterdam corridors across four scoring methods.
# =============================================================================

elif page == "🛳️ Route Builder":

    st.title("Route Builder & Comparison")
    st.markdown(
        "Compare three Vietnam → Rotterdam shipping corridors. Each scoring method encodes "
        "a different assumption about how supply chain disruptions propagate: from simple "
        "distance-weighted averages through to probabilistic cascade failures."
    )

    available_routes = sorted(routes_df['route'].unique())

    # ── Controls in the left column ───────────────────────────────────────────
    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        selected_routes = st.multiselect(
            "Select Routes",
            options=available_routes,
            default=available_routes,
            help="Select one or more routes to compare"
        )
        scoring_method = st.radio(
            "Scoring Method",
            options=list(METHOD_LABELS.keys()),
            format_func=lambda x: METHOD_LABELS[x],
            help=(
                "Simple Avg: distance-weighted mean  \n"
                "Bottleneck: blends avg with worst-case country (λ=0.4)  \n"
                "Chokepoint: applies substitutability multipliers to strategic nodes  \n"
                "Probabilistic: cascade failure model treating countries as independent risks"
            )
        )
        show_all_methods = st.checkbox("Overlay all methods", value=False,
                                       help="Show all four scoring methods on the same chart for the selected route(s)")

    with col_chart:
        if not selected_routes:
            st.warning("Select at least one route from the panel on the left.")
        else:
            df_filtered = routes_df[routes_df['route'].isin(selected_routes)]

            if show_all_methods and len(selected_routes) == 1:
                # When a single route is chosen and overlay is on, show all four
                # methods as separate lines — useful for explaining methodology divergence
                df_melt = df_filtered.melt(
                    id_vars='route',
                    value_vars=list(METHOD_LABELS.keys()),
                    var_name='method',
                    value_name='score'
                )
                df_melt['method'] = df_melt['method'].map(METHOD_LABELS)
                fig_ts = px.line(
                    df_melt,
                    x='year', y='score', color='method',
                    markers=True,
                    labels={'score': 'Risk Score (0–100)', 'year': 'Year', 'method': 'Method'},
                    title=f"{selected_routes[0]} — All Scoring Methods Over Time"
                )
            else:
                # Default: one chosen method, multiple routes
                fig_ts = px.line(
                    df_filtered,
                    x='year',
                    y=scoring_method,
                    color='route',
                    markers=True,
                    labels={scoring_method: 'Risk Score (0–100)', 'year': 'Year'},
                    title=f"Route Risk Over Time — {METHOD_LABELS[scoring_method]}"
                )

            fig_ts.update_layout(
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    # ── Side-by-side method comparison for latest year ────────────────────────
    st.subheader(f"All Four Methods — Latest Year ({int(routes_df['year'].max())})")
    st.markdown(
        "This chart reveals where the four methods agree (low analytical controversy) and "
        "where they diverge (high sensitivity to risk model assumptions). Divergence is the "
        "most interesting analytical finding — it tells you which routes are sensitive to "
        "your choice of risk philosophy."
    )

    df_latest = routes_df[
        (routes_df['route'].isin(selected_routes)) &
        (routes_df['year'] == routes_df['year'].max())
    ]

    if not df_latest.empty:
        df_melt_latest = df_latest.melt(
            id_vars='route',
            value_vars=list(METHOD_LABELS.keys()),
            var_name='Method',
            value_name='Risk Score'
        )
        df_melt_latest['Method'] = df_melt_latest['Method'].map(METHOD_LABELS)

        fig_compare = px.bar(
            df_melt_latest,
            x='Method',
            y='Risk Score',
            color='route',
            barmode='group',
            labels={'Risk Score': 'Risk Score (0–100)'},
            title="Method Comparison — Latest Year"
        )
        fig_compare.update_layout(height=340, legend_title_text='Route')
        st.plotly_chart(fig_compare, use_container_width=True)

    # ── Raw data table (collapsed by default) ─────────────────────────────────
    with st.expander("View Raw Route Scores Table"):
        display_df = (
            routes_df[routes_df['route'].isin(selected_routes)]
            .sort_values(['route', 'year'], ascending=[True, False])
            .rename(columns=METHOD_LABELS)
            .reset_index(drop=True)
        )
        st.dataframe(
            display_df.style.format({v: '{:.1f}' for v in METHOD_LABELS.values()}),
            use_container_width=True
        )


# =============================================================================
# PAGE 3 — COUNTRY DEEP DIVE
# Risk trajectory, radar profile, YoY change, and optional LSTM forecast overlay.
# =============================================================================

elif page == "🔍 Country Deep Dive":

    st.title("Country Deep Dive")
    st.markdown(
        "Drill into any of the 193 countries in the dataset. The risk trajectory shows all "
        "three composite score stages. The radar profile decomposes the latest year's score "
        "into its five underlying probability-of-disruption dimensions."
    )

    countries       = sorted(df['country'].unique())
    default_country = 'Vietnam' if 'Vietnam' in countries else countries[0]
    selected_country = st.selectbox(
        "Select a Country",
        options=countries,
        index=countries.index(default_country)
    )

    df_country = df[df['country'] == selected_country].sort_values('year')
    latest     = df_country.iloc[-1]   # Most recent year's row

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.subheader(f"{selected_country} — Snapshot ({int(latest['year'])})")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CCRS Final",  f"{latest['ccrs_final']:.1f}")
    c2.metric("Risk Tier",   latest['risk_tier'])
    c3.metric(
        "Regime",
        latest['risk_regime'].replace('_', ' ').title()
    )
    c4.metric(
        "Drawdown",
        f"{latest['ccrs_drawdown']:.1f}",
        help="How far the current score sits above the country's all-time historical minimum. "
             "A high drawdown means the country is structurally worse than it has ever been."
    )
    c5.metric(
        "5Y Volatility",
        f"{latest['ccrs_vol_5y']:.2f}",
        help="Rolling 5-year standard deviation of CCRS. Higher = more unstable risk environment. "
             "A country can have low absolute risk but high volatility — still a procurement concern."
    )

    col_left, col_right = st.columns(2)

    # ── Risk trajectory: all three composite score stages over time ───────────
    with col_left:
        fig_traj = go.Figure()

        # CCRS Final is the definitive score — plotted boldest
        fig_traj.add_trace(go.Scatter(
            x=df_country['year'],
            y=df_country['ccrs_final'],
            name='CCRS Final (with SCCR impact)',
            line=dict(color='#F44336', width=2.5),
            mode='lines+markers',
            marker=dict(size=5)
        ))
        # CCRS Compound shows the effect of dimension compounding before SCCR
        fig_traj.add_trace(go.Scatter(
            x=df_country['year'],
            y=df_country['ccrs_compound'],
            name='CCRS Compound',
            line=dict(color='#FF9800', width=1.5, dash='dot'),
            mode='lines'
        ))
        # Base CCRS shows the raw weighted composite before any amplification
        fig_traj.add_trace(go.Scatter(
            x=df_country['year'],
            y=df_country['ccrs'],
            name='CCRS Base',
            line=dict(color='#9E9E9E', width=1, dash='dash'),
            mode='lines'
        ))

        # LSTM forecast overlay — only rendered if lstm_forecasts.csv exists
        if forecasts_df is not None:
            fc = forecasts_df[forecasts_df['country'] == selected_country]
            if not fc.empty:
                fig_traj.add_trace(go.Scatter(
                    x=fc['year'],
                    y=fc['ccrs_forecast'],
                    name='LSTM Forecast',
                    line=dict(color='#9C27B0', width=2, dash='dash'),
                    mode='lines'
                ))
                # Vertical dashed line marking where historical data ends and forecast begins
                last_hist_year = int(df_country['year'].max())
                fig_traj.add_vline(
                    x=last_hist_year + 0.5,
                    line_dash='dot',
                    line_color='#9C27B0',
                    opacity=0.5,
                    annotation_text="Forecast →",
                    annotation_position="top"
                )
                # Confidence interval shading
                if 'ccrs_forecast_hi' in fc.columns and 'ccrs_forecast_lo' in fc.columns:
                    fig_traj.add_trace(go.Scatter(
                        x=pd.concat([fc['year'], fc['year'][::-1]]),
                        y=pd.concat([fc['ccrs_forecast_hi'], fc['ccrs_forecast_lo'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(156,39,176,0.1)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='95% Confidence Interval',
                        showlegend=True
                    ))

        fig_traj.update_layout(
            title=f"{selected_country} — Risk Trajectory",
            xaxis_title='Year',
            yaxis_title='CCRS Score (0–100)',
            yaxis_range=[0, 100],
            height=380,
            legend=dict(orientation='h', yanchor='bottom', y=-0.35)
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    # ── Radar chart: five probability-layer dimensions for latest year ─────────
    with col_right:
        dims   = list(DIM_LABELS.keys())
        labels = list(DIM_LABELS.values())
        values = [float(latest[d]) for d in dims]

        # Close the polygon by repeating the first value
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor='rgba(244,67,54,0.15)',
            line=dict(color='#F44336', width=2),
            name=selected_country
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=9)
                )
            ),
            title=f"Risk Dimension Profile — {int(latest['year'])}",
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Year-over-year change bar chart ───────────────────────────────────────
    st.subheader("Year-over-Year Risk Change")
    df_yoy = df_country.dropna(subset=['ccrs_yoy']).copy()
    df_yoy['direction'] = df_yoy['ccrs_yoy'].apply(
        lambda x: 'Increasing Risk' if x > 0 else 'Decreasing Risk'
    )
    fig_yoy = px.bar(
        df_yoy,
        x='year',
        y='ccrs_yoy',
        color='direction',
        color_discrete_map={
            'Increasing Risk': '#F44336',
            'Decreasing Risk': '#4CAF50'
        },
        labels={'ccrs_yoy': 'Annual Change in CCRS', 'year': 'Year'},
        title=f"{selected_country} — Annual Risk Score Change (CCRS YoY)"
    )
    fig_yoy.add_hline(y=0, line_dash='solid', line_color='#333', line_width=1)
    fig_yoy.update_layout(height=260, showlegend=True, legend_title_text='')
    st.plotly_chart(fig_yoy, use_container_width=True)

    # ── SCCR context box ──────────────────────────────────────────────────────
    with st.expander(f"What is the SCCR score for {selected_country}?"):
        st.markdown(
            f"**SCCR (Supply Chain Concentration Risk): {latest['sccr']:.1f}**  \n\n"
            "SCCR measures how much global trade would be disrupted if this country were taken "
            "offline. It is applied as a post-composite impact multiplier on top of the "
            "probability-of-disruption CCRS Compound score. Countries like Singapore and Egypt "
            "can have relatively low fragility scores but very high SCCR — meaning a disruption "
            "is unlikely but catastrophic if it occurs.  \n\n"
            f"SCCR Multiplier applied: **{latest['sccr_multiplier']:.3f}×**"
        )

    if forecasts_df is None:
        st.info(
            "💡 **LSTM forecast overlay is not enabled.** To add it, make sure "
            "`lstm_forecasts.csv` is in your repository root (it should already be there — "
            "check your SCRM folder). The app expects columns: `iso3`, `country`, `year`, "
            "`ccrs_forecast`, `ccrs_forecast_hi`, `ccrs_forecast_lo`."
        )


# =============================================================================
# PAGE 4 — PORTFOLIO RISK SUMMARY
# Cross-country landscape: scatter quadrant, regime pie, drawdown ranking,
# top movers. The scatter chart is the signature "investment-style" visual.
# =============================================================================

elif page == "📊 Portfolio Summary":

    st.title("Portfolio Risk Summary")
    st.markdown(
        "Cross-country risk landscape viewed through an investment-research lens. "
        "The scatter chart maps Risk Level against Risk Instability — the two axes "
        "that matter most for procurement portfolio decisions."
    )

    years = sorted(df['year'].unique())
    selected_year = st.slider(
        "Year",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=int(max(years)),
        key='portfolio_year_slider'
    )

    df_year = df[
        (df['year'] == selected_year) &
        df['ccrs_vol_5y'].notna()
    ].copy()

    # ── Risk / Volatility scatter ─────────────────────────────────────────────
    # This is the signature chart: think of it as a risk-return scatter from
    # equity research, but the axes are risk level (x) and risk instability (y).
    # Quadrant II (high risk, high volatility) = most dangerous for procurement.
    # Quadrant IV (low risk, low volatility) = safest sourcing environments.
    col1, col2 = st.columns([3, 2])

    with col1:
        median_risk = df_year['ccrs_final'].median()
        median_vol  = df_year['ccrs_vol_5y'].median()

        fig_scatter = px.scatter(
            df_year,
            x='ccrs_final',
            y='ccrs_vol_5y',
            color='risk_tier',
            color_discrete_map=TIER_COLORS,
            hover_name='country',
            hover_data={
                'ccrs_final':  ':.1f',
                'ccrs_vol_5y': ':.2f',
                'risk_regime': True,
                'sccr':        ':.1f',
            },
            labels={
                'ccrs_final':  'CCRS Final Score (Risk Level →)',
                'ccrs_vol_5y': '5Y Volatility (Risk Instability ↑)',
            },
            title=f"Risk Level vs Risk Instability — {selected_year}",
        )

        # Median reference lines create the four quadrants
        fig_scatter.add_vline(x=median_risk, line_dash='dash', line_color='#ccc', opacity=0.7)
        fig_scatter.add_hline(y=median_vol,  line_dash='dash', line_color='#ccc', opacity=0.7)

        # Quadrant corner labels — mimics investment research style
        x_max = df_year['ccrs_final'].quantile(0.97)
        y_max = df_year['ccrs_vol_5y'].quantile(0.97)
        x_min = df_year['ccrs_final'].quantile(0.03)
        y_min = df_year['ccrs_vol_5y'].quantile(0.03)

        for text, x, y, color in [
            ("High Risk / High Volatility", x_max * 0.85, y_max * 0.92, '#F44336'),
            ("Low Risk / Low Volatility",   x_min * 1.15, y_min * 1.20, '#2196F3'),
            ("High Risk / Low Volatility",  x_max * 0.85, y_min * 1.20, '#FF9800'),
            ("Low Risk / High Volatility",  x_min * 1.15, y_max * 0.92, '#9E9E9E'),
        ]:
            fig_scatter.add_annotation(
                x=x, y=y, text=text, showarrow=False,
                font=dict(color=color, size=9), opacity=0.75
            )

        fig_scatter.update_layout(height=430, legend_title_text='Risk Tier')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # ── Regime distribution doughnut ──────────────────────────────────────
        regime_counts = df_year['risk_regime'].value_counts().reset_index()
        regime_counts.columns = ['Regime', 'Count']
        fig_regime = px.pie(
            regime_counts,
            names='Regime',
            values='Count',
            color='Regime',
            color_discrete_map=REGIME_COLORS,
            title=f"Risk Regime Distribution — {selected_year}",
            hole=0.45
        )
        fig_regime.update_layout(height=210, margin=dict(t=40, b=0))
        st.plotly_chart(fig_regime, use_container_width=True)

        # ── Risk tier count bars ───────────────────────────────────────────────
        tier_order  = ['Low', 'Medium-Low', 'Medium-High', 'High']
        tier_counts = (
            df_year['risk_tier']
            .value_counts()
            .reindex(tier_order)
            .fillna(0)
            .reset_index()
        )
        tier_counts.columns = ['Tier', 'Count']
        fig_tiers = px.bar(
            tier_counts,
            x='Count',
            y='Tier',
            orientation='h',
            color='Tier',
            color_discrete_map=TIER_COLORS,
            text='Count',
            title="Countries by Risk Tier"
        )
        fig_tiers.update_traces(textposition='outside')
        fig_tiers.update_layout(showlegend=False, height=200, margin=dict(t=40))
        st.plotly_chart(fig_tiers, use_container_width=True)

    # ── Top movers: biggest YoY risk increases and decreases ──────────────────
    st.subheader(f"Biggest Risk Movers — {selected_year}")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🔴 Largest Risk Increases (YoY)**")
        risers = (
            df_year
            .nlargest(8, 'ccrs_yoy')
            [['country', 'ccrs_final', 'ccrs_yoy', 'risk_tier', 'risk_regime']]
            .reset_index(drop=True)
        )
        risers.index += 1
        risers.columns = ['Country', 'CCRS Final', 'YoY Δ', 'Tier', 'Regime']
        st.dataframe(
            risers.style.format({'CCRS Final': '{:.1f}', 'YoY Δ': '+{:.2f}'}),
            use_container_width=True
        )

    with col_b:
        st.markdown("**🟢 Largest Risk Decreases (YoY)**")
        fallers = (
            df_year
            .nsmallest(8, 'ccrs_yoy')
            [['country', 'ccrs_final', 'ccrs_yoy', 'risk_tier', 'risk_regime']]
            .reset_index(drop=True)
        )
        fallers.index += 1
        fallers.columns = ['Country', 'CCRS Final', 'YoY Δ', 'Tier', 'Regime']
        st.dataframe(
            fallers.style.format({'CCRS Final': '{:.1f}', 'YoY Δ': '{:.2f}'}),
            use_container_width=True
        )

    # ── Structural deterioration: countries furthest from their historical best ─
    st.subheader("Structural Deterioration — Countries Furthest From Historical Best")
    st.markdown(
        "Drawdown measures how far a country's current CCRS sits above its all-time historical "
        "minimum. A high drawdown means the country is in structurally worse shape than it has "
        "ever been — even if its absolute score is not the highest globally. This is the "
        "supply-chain equivalent of a portfolio drawdown metric."
    )

    drawdown_top = (
        df_year
        .nlargest(12, 'ccrs_drawdown')
        [['country', 'ccrs_final', 'ccrs_drawdown', 'risk_regime', 'risk_tier']]
        .reset_index(drop=True)
    )

    fig_drawdown = px.bar(
        drawdown_top,
        x='ccrs_drawdown',
        y='country',
        orientation='h',
        color='ccrs_final',
        color_continuous_scale='RdYlBu_r',
        range_color=(0, 100),
        hover_data={'risk_tier': True, 'risk_regime': True},
        labels={
            'ccrs_drawdown': 'Drawdown from Historical Best',
            'country':       'Country',
            'ccrs_final':    'CCRS Final Score'
        },
        title=f"Top 12 Countries by Risk Drawdown — {selected_year}"
    )
    fig_drawdown.update_layout(
        height=380,
        yaxis=dict(autorange='reversed'),
        coloraxis_colorbar=dict(title="CCRS Final")
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)
