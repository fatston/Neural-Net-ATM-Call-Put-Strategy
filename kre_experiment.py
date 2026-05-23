#!/usr/bin/env python3
"""Toy neural-network experiment for predicting KRE's next daily direction."""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf


TICKER = "KRE"


@dataclass
class StandardScaler:
    """Small train-only scaler to avoid leaking test data information."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, data: pd.DataFrame) -> "StandardScaler":
        std = data.std().replace(0, 1)
        return cls(mean=data.mean(), std=std)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data - self.mean) / self.std


class DirectionNet(nn.Module):
    """A deliberately small feed-forward neural network."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def download_history(start: str) -> pd.DataFrame:
    """Download adjusted daily KRE history from Yahoo Finance."""
    data = yf.download(TICKER, start=start, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No {TICKER} data downloaded. Check your network connection.")

    # yfinance can return multi-index columns in some versions/settings.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[["Open", "High", "Low", "Close", "Volume"]].dropna()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute a simple RSI using rolling average gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create historical-only features and tomorrow-up target."""
    df = data.copy()

    # All features below are known at today's close.
    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)
    df["rolling_volatility"] = df["daily_return"].rolling(20).std()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_distance"] = df["Close"] / df["ma_20"] - 1
    df["volume_change"] = df["Volume"].pct_change()
    df["rsi_14"] = compute_rsi(df["Close"], 14)

    # Target and forward return use tomorrow's close, so they are labels only.
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(float)
    df["forward_return"] = df["Close"].shift(-1) / df["Close"] - 1

    return df


def chronological_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date order, never randomly."""
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def date_window_split(
    df: pd.DataFrame,
    train_fraction: float,
    test_start: str | None,
    test_end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use an explicit test window when provided, otherwise use a standard split."""
    if test_start is None:
        return chronological_split(df, train_fraction)

    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end) if test_end else df.index.max()
    train_df = df.loc[df.index < start].copy()
    test_df = df.loc[(df.index >= start) & (df.index <= end)].copy()

    if train_df.empty:
        raise ValueError("No training rows before --test-start. Pick an earlier --start or later --test-start.")
    if test_df.empty:
        raise ValueError("No test rows in the requested date window.")

    return train_df, test_df


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> DirectionNet:
    """Train the neural network with binary cross entropy."""
    torch.manual_seed(seed)
    model = DirectionNet(input_size=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    features = torch.tensor(x_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return model


def predict_probabilities(model: DirectionNet, features: np.ndarray) -> np.ndarray:
    """Return P(tomorrow close > today close)."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= 0.5).astype(int)
    return float((predictions == y_true).mean())


def confusion_matrix(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    predictions = (probabilities >= 0.5).astype(int)
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true.astype(int), predictions):
        matrix[actual, predicted] += 1
    return matrix


def run_backtest(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    iv_multiplier: float,
    option_cost_bps: float,
) -> pd.DataFrame:
    """Compare buy-and-hold with NN and simple option-buying strategies."""
    results = test_df[["forward_return"]].copy()
    results["probability"] = probabilities

    results["buy_hold_return"] = results["forward_return"]
    results["buy_hold_equity"] = (1 + results["buy_hold_return"]).cumprod()
    results["nn_position"] = (results["probability"] >= 0.50).astype(int)
    results["nn_strategy_return"] = results["nn_position"] * results["forward_return"]
    results["nn_strategy_equity"] = (1 + results["nn_strategy_return"]).cumprod()
    results["daily_call_equity"] = close_to_close_call_equity_curve(test_df, iv_multiplier, option_cost_bps)
    results["nn_option_equity"] = close_to_close_nn_option_equity_curve(
        test_df,
        probabilities,
        iv_multiplier,
        option_cost_bps,
    )
    return results


def normal_cdf(value: float) -> float:
    """Normal CDF without adding scipy as a dependency."""
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def black_scholes_call_price(spot: float, strike: float, volatility: float, years: float) -> float:
    """Price a simple European call with zero rates/dividends."""
    volatility = max(float(volatility), 0.01)
    years = max(float(years), 1 / 252)
    denominator = volatility * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * volatility**2 * years) / denominator
    d2 = d1 - denominator
    return spot * normal_cdf(d1) - strike * normal_cdf(d2)


def black_scholes_put_price(spot: float, strike: float, volatility: float, years: float) -> float:
    """Price a simple European put with zero rates/dividends."""
    volatility = max(float(volatility), 0.01)
    years = max(float(years), 1 / 252)
    denominator = volatility * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * volatility**2 * years) / denominator
    d2 = d1 - denominator
    return strike * normal_cdf(-d2) - spot * normal_cdf(-d1)


def close_to_close_call_equity_curve(
    test_df: pd.DataFrame,
    iv_multiplier: float,
    option_cost_bps: float,
    allocation: float = 0.01,
) -> pd.Series:
    """Buy a 1-day ATM call close-to-close using 1% of the portfolio.

    This is only a rough proxy because it uses Black-Scholes and realized
    volatility instead of historical option-chain prices. The IV multiplier and
    cost assumption make the fill less generous than pure realized volatility.
    """
    equity = []
    portfolio = 1.0
    premium_multiplier = 1 + option_cost_bps / 10_000

    for _, row in test_df.iterrows():
        spot = float(row["Close"])
        next_close = spot * (1 + float(row["forward_return"]))
        volatility = float(row["rolling_volatility"] * math.sqrt(252) * iv_multiplier)
        fair_premium = black_scholes_call_price(spot=spot, strike=spot, volatility=volatility, years=1 / 252)
        paid_premium = fair_premium * premium_multiplier
        payoff = max(next_close - spot, 0)

        option_value_ratio = payoff / paid_premium if paid_premium > 0 else 0
        portfolio *= (1 - allocation) + allocation * option_value_ratio
        equity.append(portfolio)

    return pd.Series(equity, index=test_df.index)


def close_to_close_nn_option_equity_curve(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    iv_multiplier: float,
    option_cost_bps: float,
    allocation: float = 0.01,
    direction_threshold: float = 0.50,
) -> pd.Series:
    """Buy a 1-day ATM call or put close-to-close based on the NN's direction."""
    equity = []
    portfolio = 1.0
    premium_multiplier = 1 + option_cost_bps / 10_000

    for (_, row), probability in zip(test_df.iterrows(), probabilities):
        spot = float(row["Close"])
        next_close = spot * (1 + float(row["forward_return"]))
        volatility = float(row["rolling_volatility"] * math.sqrt(252) * iv_multiplier)

        if probability >= direction_threshold:
            fair_premium = black_scholes_call_price(spot=spot, strike=spot, volatility=volatility, years=1 / 252)
            payoff = max(next_close - spot, 0)
        else:
            fair_premium = black_scholes_put_price(spot=spot, strike=spot, volatility=volatility, years=1 / 252)
            payoff = max(spot - next_close, 0)

        paid_premium = fair_premium * premium_multiplier
        option_value_ratio = payoff / paid_premium if paid_premium > 0 else 0
        portfolio *= (1 - allocation) + allocation * option_value_ratio
        equity.append(portfolio)

    return pd.Series(equity, index=test_df.index)


def max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown as a negative fraction."""
    return float((equity / equity.cummax() - 1).min())


def sharpe_like(returns: pd.Series) -> float:
    """Simple annualized daily Sharpe-style statistic."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * math.sqrt(252))


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate for a dated equity series."""
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    return float(equity.iloc[-1] ** (1 / years) - 1)


def print_strategy_stats(name: str, equity: pd.Series) -> None:
    """Print a compact risk summary for one equity curve."""
    returns = equity.pct_change().dropna()
    print(
        f"{name}: final={equity.iloc[-1]:.3f}, "
        f"CAGR={cagr(equity):.1%}, "
        f"maxDD={max_drawdown(equity):.1%}, "
        f"Sharpe-ish={sharpe_like(returns):.2f}, "
        f"worst_day={returns.min():.1%}"
    )


def latest_signal(probability: float) -> str:
    """Convert the latest probability into a simple buy/hold/sell-style label."""
    if probability > 0.55:
        return "BUY / LONG for tomorrow"
    if probability < 0.45:
        return "SELL / STAY FLAT for tomorrow"
    return "HOLD / NO STRONG EDGE"


def plot_equity_curve(backtest: pd.DataFrame, output_path: str) -> None:
    """Save a strategy-vs-buy-and-hold equity curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(backtest.index, backtest["buy_hold_equity"], label="Buy and hold KRE")
    plt.plot(backtest.index, backtest["nn_strategy_equity"], label="NN decides long/flat")
    plt.plot(backtest.index, backtest["daily_call_equity"], label="Close-to-close 1% ATM call buyer")
    plt.plot(backtest.index, backtest["nn_option_equity"], label="Close-to-close NN 1% ATM call/put buyer")
    plt.title("KRE Toy Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy PyTorch model on KRE daily data.")
    parser.add_argument("--start", default="2000-01-01", help="First date to download from yfinance.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological training fraction.")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--iv-multiplier", type=float, default=1.25, help="Multiplier applied to realized volatility for option pricing.")
    parser.add_argument("--option-cost-bps", type=float, default=10.0, help="Extra option premium paid as execution cost in basis points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for PyTorch.")
    parser.add_argument("--test-start", default=None, help="Optional first date for the backtest window.")
    parser.add_argument("--test-end", default=None, help="Optional final date for the backtest window.")
    parser.add_argument("--plot", default="kre_equity_curve.png", help="Output path for equity curve plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = download_history(args.start)
    featured = make_features(raw)

    feature_columns = [
        "daily_return",
        "return_5d",
        "return_10d",
        "return_20d",
        "rolling_volatility",
        "ma_distance",
        "volume_change",
        "rsi_14",
    ]

    # Drop rows where features or labels are not yet available.
    model_data = featured.dropna(subset=feature_columns + ["target", "forward_return"]).copy()
    train_df, test_df = date_window_split(model_data, args.train_fraction, args.test_start, args.test_end)

    scaler = StandardScaler.fit(train_df[feature_columns])
    x_train = scaler.transform(train_df[feature_columns]).to_numpy(dtype=np.float32)
    x_test = scaler.transform(test_df[feature_columns]).to_numpy(dtype=np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.int64)
    y_test = test_df["target"].to_numpy(dtype=np.int64)

    model = train_model(x_train, y_train, args.epochs, args.learning_rate, args.seed)

    train_prob = predict_probabilities(model, x_train)
    test_prob = predict_probabilities(model, x_test)

    backtest = run_backtest(test_df, test_prob, args.iv_multiplier, args.option_cost_bps)
    plot_equity_curve(backtest, args.plot)

    # The latest row may not have a target yet, but it has today's known features.
    latest_row = featured.dropna(subset=feature_columns).iloc[-1:]
    latest_features = scaler.transform(latest_row[feature_columns]).to_numpy(dtype=np.float32)
    latest_probability = float(predict_probabilities(model, latest_features)[0])

    print(f"Rows used: train={len(train_df):,}, test={len(test_df):,}")
    print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print(f"Train accuracy: {accuracy(y_train, train_prob):.3f}")
    print(f"Test accuracy:  {accuracy(y_test, test_prob):.3f}")
    print("Confusion matrix on test set [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, test_prob))
    print()
    print(f"Latest date: {latest_row.index[-1].date()}")
    print(f"Latest predicted probability of KRE closing higher tomorrow: {latest_probability:.3f}")
    print(f"Signal: {latest_signal(latest_probability)}")
    print()
    print(f"Option IV multiplier: {args.iv_multiplier:.2f}")
    print(f"Option execution cost: {args.option_cost_bps:.1f} bps of premium")
    print()
    print("Risk summary:")
    print_strategy_stats("Buy-and-hold", backtest["buy_hold_equity"])
    print_strategy_stats("NN decides long/flat", backtest["nn_strategy_equity"])
    print_strategy_stats("Close-to-close 1% ATM call", backtest["daily_call_equity"])
    print_strategy_stats("Close-to-close NN 1% ATM call/put", backtest["nn_option_equity"])
    print(f"Equity curve saved to: {args.plot}")


if __name__ == "__main__":
    main()
