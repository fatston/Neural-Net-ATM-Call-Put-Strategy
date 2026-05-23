# Neural Net ATM Call/Put Strategy

This is a small local Python experiment that trains a simple PyTorch neural
network to predict whether SPY will close higher tomorrow than it closed today.

It is intentionally simple and readable. It is an educational toy model, not
financial advice, not a trading system, and not a recommendation to buy or sell
anything.

## What It Does

- Downloads daily SPY data from Yahoo Finance with `yfinance`.
- Builds historical-only features such as returns, volatility, moving-average
  distance, volume change, and RSI.
- Creates a target where `1` means tomorrow's close is higher than today's close.
- Splits data chronologically into train and test sets.
- Trains a small neural network with PyTorch.
- Prints train accuracy, test accuracy, a confusion matrix, and the latest
  buy/hold/sell-style signal.
- Adds a rough close-to-close options toy strategy:
  - Buy a 1-day at-the-money call at today's close.
  - Settle it against tomorrow's close.
  - Spend `1%` of the current portfolio each day.
  - Price the option with a simple Black-Scholes proxy using rolling realized
    volatility times an implied-volatility multiplier, not real historical
    option-chain prices.
- Adds another rough options toy strategy:
  - Spend `1%` each day on a 1-day at-the-money call when the neural network
    predicts up.
  - Spend `1%` each day on a 1-day at-the-money put when the neural network
    predicts down.
- Saves an equity curve plot comparing buy-and-hold SPY and the options toy
  strategies.
- Prints a compact risk summary with final equity, CAGR, max drawdown,
  Sharpe-style ratio, and worst daily return.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python spy_nn_experiment.py
```

The script prints model metrics and saves:

```text
equity_curve.png
```

## Optional Arguments

```bash
python spy_nn_experiment.py --start 2010-01-01 --epochs 300
```

The option proxy is intentionally less generous than pure realized-vol pricing:

```bash
python spy_nn_experiment.py --iv-multiplier 1.25 --option-cost-bps 10
```

To train on data before a specific period and backtest only that period:

```bash
python spy_nn_experiment.py --test-start 2020-02-19 --test-end 2020-03-23 --plot equity_curve_bear.png
```

Useful options:

- `--start`: first date to download from yfinance.
- `--train-fraction`: chronological training fraction, default `0.8`.
- `--epochs`: neural-network training epochs, default `250`.
- `--iv-multiplier`: multiplier applied to rolling realized volatility when
  pricing options, default `1.25`.
- `--option-cost-bps`: extra premium paid as execution cost, default `10`.
- `--test-start`: optional first date for an explicit backtest window.
- `--test-end`: optional final date for an explicit backtest window.
- `--plot`: output path for the equity curve image.

## Notes On Lookahead Bias

The features use only information available by the current day's close. The
target and forward return use the next day's close, but those columns are used
only as labels and backtest returns, not as model inputs.

The scaler is fit on the training set only, then reused for the test set and the
latest signal.

## Reminder

Markets are noisy, non-stationary, and hard to predict. This project is for
learning Python, PyTorch, feature engineering, and backtest mechanics. Do not
use it as financial advice.

The close-to-close option strategies are especially simplified. Real options have bid/ask
spreads, implied volatility, early close/holiday effects, assignment/exercise
details, strike availability, and liquidity constraints. Treat that curve as a
toy thought experiment, not a realistic options backtest.
