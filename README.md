# Financial Market Analytics Dashboard

An interactive Python/Dash dashboard for exploratory analysis of stocks and cryptocurrencies — volatility, correlation, moving averages, and risk-adjusted return metrics, powered by live market data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/dash-plotly-informational)

## Features

- **Price & Moving Averages** — price charts with 20/50/200-day moving averages for any selected asset
- **Volatility Analysis** — 30-day rolling annualized volatility, overall volatility ranking across assets, and visual inspection of volatility clustering
- **Correlation Analysis** — static correlation heatmap across all selected assets, plus a rolling 90-day correlation view between any two assets, showing how relationships shift over time (e.g. crypto vs. equities during market stress)
- **Risk & Return Stats** — annualized return, annualized volatility, Sharpe ratio, skew, and excess kurtosis per asset, plus drawdown charts and return distribution histograms

## Screenshot

*(Add a screenshot here — see "Adding a screenshot" below)*

## Tech Stack

- [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/python/) — interactive web app and charting
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation and statistics
- [yfinance](https://github.com/ranaroussi/yfinance) — free historical market data from Yahoo Finance
- [SciPy](https://scipy.org/) — skew/kurtosis calculations

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/<your-username>/financial-eda-dashboard.git
cd financial-eda-dashboard

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

## Usage

1. Select one or more tickers (stocks like `AAPL`, or crypto like `BTC-USD`)
2. Choose a date range and return type (log or simple returns)
3. Click **Load / Refresh Data**
4. Explore the four tabs: Price & Moving Averages, Volatility, Correlation, and Return Stats

## Methodology Notes

- **Log returns** are used by default since they're additive over time and better approximate normality, which most risk statistics assume
- **Volatility** is annualized using the standard `σ_daily × √252` convention (252 trading days/year)
- **Sharpe ratio** is computed without a risk-free rate offset (i.e. `mean return / volatility`), so it should be read as a relative risk-adjusted return measure rather than an absolute one
- **Correlation** is computed on returns, not raw prices, to avoid spurious correlation from shared upward price trends

## Project Structure

```
.
├── app.py              # Main Dash application
├── requirements.txt     # Python dependencies
└── README.md
```

## Possible Extensions

- Add a risk-free rate input for a proper Sharpe ratio
- Support portfolio-level analysis (weighted combinations of assets)
- Add VaR (Value at Risk) / CVaR calculations
- Deploy to a hosting service (e.g. Render, Railway) for a live public demo

## License

MIT — feel free to use or adapt this project.
