from src.config import load_config
from src.data import load_data
from src.indicators import add_indicators
from src.features import (
    add_aoi_features,
    clean_features
)
from src.labels import create_labels
from src.ml.walk_forward import walk_forward
from src.ml.backtest import backtest_ml
from src.metrics import calculate_metrics


def main():

    config = load_config()

    path = config["data"]["parquet"]

    print("Loading data...")

    df = load_data(path)

    print(
        f"Loaded {df.height:,} candles"
    )

    print("Calculating indicators...")

    df = add_indicators(df)

    print("Calculating AOI...")

    df = add_aoi_features(df)

    print("Creating ML labels...")

    df = create_labels(
        df,
        horizon=
            config["ml"]["horizon"]
    )

    df = clean_features(df)

    print(
        f"Training rows: "
        f"{df.height:,}"
    )

    print(
        "Running walk-forward "
        "training..."
    )

    results = walk_forward(
        df,
        config
    )

    print(
        f"\nPredictions: "
        f"{len(results):,}"
    )

    print(
        "Running ML backtest..."
    )

    strategy = config["strategy"]
    ml = config["ml"]

    trades, balance = backtest_ml(

        results,

        initial_balance=10_000,

        risk_per_trade=
            strategy["risk_per_trade"],

        sl_atr=
            strategy["atr_sl_multiplier"],

        tp_atr=
            strategy["atr_tp_multiplier"],

        cost_pips=
            strategy["transaction_cost_pips"],

        probability_long=
            ml["probability_long"],

        probability_short=
            ml["probability_short"]
    )

    metrics = calculate_metrics(
        trades,
        10_000
    )

    print("\n===== ML STRATEGY =====")

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        f"\nFinal balance: "
        f"{balance:.2f}"
    )


if __name__ == "__main__":
    main()