"""
Financial Market EDA Dashboard
--------------------------------
Interactive Dash app for exploring stock/crypto data:
- Price + moving averages
- Rolling & annualized volatility
- Static & rolling correlation between assets
- Return distribution stats (skew, kurtosis, Sharpe ratio)

Run with:  python app.py
Then open: http://127.0.0.1:8050
"""

import io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRADING_DAYS = 252

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]

app = dash.Dash(__name__, title="Financial EDA Dashboard")
server = app.server  # for deployment (gunicorn etc.)

# Several components (ma-ticker, ma-graph, corr-a, corr-b, rolling-corr-graph)
# are only created dynamically inside tab-content after data loads, so Dash
# can't validate them against the initial layout at startup. Suppress that check.
app.config.suppress_callback_exceptions = True


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H2("Financial Market EDA Dashboard"),
        html.P("Volatility, correlation, and moving average analysis for stocks & crypto.",
               style={"color": "#555"}),

        # ---- Controls ----
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "alignItems": "flex-end",
                   "marginBottom": "20px", "padding": "15px", "backgroundColor": "#f7f7f9",
                   "borderRadius": "8px"},
            children=[
                html.Div([
                    html.Label("Tickers (stocks: AAPL, crypto: BTC-USD, etc.)"),
                    dcc.Dropdown(
                        id="ticker-input",
                        options=[{"label": t, "value": t} for t in
                                 DEFAULT_TICKERS + ["NVDA", "AMZN", "SOL-USD", "TSLA"]],
                        value=DEFAULT_TICKERS,
                        multi=True,
                        style={"width": "420px"},
                    ),
                ]),
                html.Div([
                    html.Label("Date range"),
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date=(pd.Timestamp.today() - pd.DateOffset(years=3)).date(),
                        end_date=pd.Timestamp.today().date(),
                    ),
                ]),
                html.Div([
                    html.Label("Return type"),
                    dcc.RadioItems(
                        id="return-type",
                        options=[{"label": "Log returns", "value": "log"},
                                 {"label": "Simple returns", "value": "simple"}],
                        value="log",
                        inline=True,
                    ),
                ]),
                html.Button("Load / Refresh Data", id="load-button", n_clicks=0,
                            style={"height": "38px", "padding": "0 20px", "backgroundColor": "#2c6cf0",
                                   "color": "white", "border": "none", "borderRadius": "5px",
                                   "cursor": "pointer"}),
            ],
        ),

        html.Div(id="status-msg", style={"color": "#b00", "marginBottom": "10px"}),

        # Hidden store holding the downloaded price data (as JSON) so tabs can share it
        dcc.Store(id="price-store"),

        dcc.Tabs(id="tabs", value="tab-prices", children=[
            dcc.Tab(label="Price & Moving Averages", value="tab-prices"),
            dcc.Tab(label="Volatility", value="tab-vol"),
            dcc.Tab(label="Correlation", value="tab-corr"),
            dcc.Tab(label="Return Stats", value="tab-stats"),
        ]),

        html.Div(id="tab-content", style={"paddingTop": "20px"}),
    ],
)

# ---------------------------------------------------------------------------
# Data loading callback
# ---------------------------------------------------------------------------

@app.callback(
    Output("price-store", "data"),
    Output("status-msg", "children"),
    Input("load-button", "n_clicks"),
    State("ticker-input", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=False,
)
def load_data(n_clicks, tickers, start_date, end_date):
    if not tickers:
        return None, "Select at least one ticker."
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
        if isinstance(raw, pd.Series):  # single ticker returns a Series
            raw = raw.to_frame(name=tickers[0])
        raw = raw.dropna(how="all").ffill().dropna()
        if raw.empty:
            return None, "No data returned for that ticker/date combination."
        raw.index.name = "Date"
        return raw.reset_index().to_json(date_format="iso", orient="split"), ""
    except Exception as e:
        return None, f"Error loading data: {e}"


def _load_df(store_data):
    """Helper: turn stored JSON back into a DataFrame indexed by Date."""
    df = pd.read_json(io.StringIO(store_data), orient="split")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def _returns(df, kind="log"):
    if kind == "log":
        return np.log(df / df.shift(1)).dropna()
    return df.pct_change().dropna()


# ---------------------------------------------------------------------------
# Tab rendering
# ---------------------------------------------------------------------------

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("price-store", "data"),
    State("return-type", "value"),
)
def render_tab(tab, store_data, return_type):
    if not store_data:
        return html.Div("Click 'Load / Refresh Data' to begin.", style={"color": "#777"})

    df = _load_df(store_data)
    rets = _returns(df, return_type)
    tickers = df.columns.tolist()

    if tab == "tab-prices":
        return _prices_layout(df, tickers)
    elif tab == "tab-vol":
        return _volatility_layout(rets, tickers)
    elif tab == "tab-corr":
        return _correlation_layout(rets, tickers)
    elif tab == "tab-stats":
        return _stats_layout(rets, tickers)
    return html.Div()


def _prices_layout(df, tickers):
    ticker_dropdown = dcc.Dropdown(
        id="ma-ticker",
        options=[{"label": t, "value": t} for t in tickers],
        value=tickers[0],
        style={"width": "250px", "marginBottom": "15px"},
    )
    return html.Div([
        html.Label("Select asset for moving-average view:"),
        ticker_dropdown,
        dcc.Graph(id="ma-graph"),
    ])


@app.callback(
    Output("ma-graph", "figure"),
    Input("ma-ticker", "value"),
    State("price-store", "data"),
)
def update_ma_graph(ticker, store_data):
    if not store_data or not ticker:
        return go.Figure()
    df = _load_df(store_data)
    series = df[ticker]
    ma20 = series.rolling(20).mean()
    ma50 = series.rolling(50).mean()
    ma200 = series.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name="Price", line=dict(width=1, color="grey")))
    fig.add_trace(go.Scatter(x=ma20.index, y=ma20, name="MA20"))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="MA50"))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="MA200"))
    fig.update_layout(title=f"{ticker} Price with Moving Averages", template="plotly_white",
                       xaxis_title="Date", yaxis_title="Price")
    return fig


def _volatility_layout(rets, tickers):
    rolling_vol = rets.rolling(30).std() * np.sqrt(TRADING_DAYS)
    annual_vol = (rets.std() * np.sqrt(TRADING_DAYS)).sort_values(ascending=False)

    vol_fig = go.Figure()
    for t in tickers:
        vol_fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[t], name=t))
    vol_fig.update_layout(title="30-Day Rolling Annualized Volatility", template="plotly_white",
                           xaxis_title="Date", yaxis_title="Annualized Volatility")

    bar_fig = go.Figure(go.Bar(x=annual_vol.index, y=annual_vol.values))
    bar_fig.update_layout(title="Overall Annualized Volatility by Asset", template="plotly_white",
                           yaxis_title="Annualized Volatility")

    returns_fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True,
                                 subplot_titles=[f"{t} Daily Returns" for t in tickers])
    for i, t in enumerate(tickers, start=1):
        returns_fig.add_trace(go.Scatter(x=rets.index, y=rets[t], name=t, showlegend=False), row=i, col=1)
    returns_fig.update_layout(height=180 * len(tickers), template="plotly_white",
                               title="Volatility Clustering (Raw Daily Returns)")

    return html.Div([
        dcc.Graph(figure=vol_fig),
        dcc.Graph(figure=bar_fig),
        dcc.Graph(figure=returns_fig),
    ])


def _correlation_layout(rets, tickers):
    corr = rets.corr()
    heatmap = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    heatmap.update_layout(title="Static Return Correlation Matrix", template="plotly_white")

    rolling_controls = html.Div([
        html.Label("Rolling correlation between:"),
        dcc.Dropdown(id="corr-a", options=[{"label": t, "value": t} for t in tickers],
                      value=tickers[0], style={"width": "200px", "display": "inline-block"}),
        html.Span(" vs ", style={"margin": "0 10px"}),
        dcc.Dropdown(id="corr-b", options=[{"label": t, "value": t} for t in tickers],
                      value=tickers[-1], style={"width": "200px", "display": "inline-block"}),
    ], style={"marginTop": "20px", "marginBottom": "10px"})

    return html.Div([
        dcc.Graph(figure=heatmap),
        rolling_controls,
        dcc.Graph(id="rolling-corr-graph"),
    ])


@app.callback(
    Output("rolling-corr-graph", "figure"),
    Input("corr-a", "value"),
    Input("corr-b", "value"),
    Input("price-store", "data"),
    State("return-type", "value"),
)
def update_rolling_corr(a, b, store_data, return_type):
    if not store_data or not a or not b:
        return go.Figure()
    df = _load_df(store_data)
    rets = _returns(df, return_type)
    roll = rets[a].rolling(90).corr(rets[b])
    fig = go.Figure(go.Scatter(x=roll.index, y=roll, name=f"{a} vs {b}"))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(title=f"90-Day Rolling Correlation: {a} vs {b}", template="plotly_white",
                       yaxis_title="Correlation", yaxis_range=[-1, 1])
    return fig


def _stats_layout(rets, tickers):
    rows = []
    for t in tickers:
        r = rets[t]
        annual_return = r.mean() * TRADING_DAYS
        annual_vol = r.std() * np.sqrt(TRADING_DAYS)
        sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
        rows.append({
            "Ticker": t,
            "Annualized Return": f"{annual_return:.2%}",
            "Annualized Volatility": f"{annual_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Skew": f"{stats.skew(r):.2f}",
            "Excess Kurtosis": f"{stats.kurtosis(r):.2f}",
        })
    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_cell={"textAlign": "center", "padding": "8px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
    )

    # Drawdown chart
    cumulative = (1 + rets).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    dd_fig = go.Figure()
    for t in tickers:
        dd_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown[t], name=t))
    dd_fig.update_layout(title="Drawdown (% decline from running peak)", template="plotly_white",
                          yaxis_tickformat=".0%")

    # Histogram of returns
    hist_fig = go.Figure()
    for t in tickers:
        hist_fig.add_trace(go.Histogram(x=rets[t], name=t, opacity=0.5, nbinsx=60))
    hist_fig.update_layout(barmode="overlay", title="Return Distribution", template="plotly_white")

    return html.Div([
        html.H4("Risk & Return Summary"),
        table,
        dcc.Graph(figure=dd_fig, style={"marginTop": "25px"}),
        dcc.Graph(figure=hist_fig),
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
