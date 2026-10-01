import polars as pl


def generate_signals(df, config):

    technical = config["technical"]
    strategy = config["strategy"]

    long_condition = (
        (pl.col("sma50") > pl.col("sma100"))
        &
        (pl.col("macd") > pl.col("macd_signal"))
        &
        (pl.col("rsi") >= technical["rsi_long_min"])
        &
        (pl.col("rsi") <= technical["rsi_long_max"])
        &
        (pl.col("bb_position") <= technical["bollinger_distance"])
        &
        (pl.col("atr_ratio") >= strategy["min_atr_ratio"])
    )

    short_condition = (
        (pl.col("sma50") < pl.col("sma100"))
        &
        (pl.col("macd") < pl.col("macd_signal"))
        &
        (pl.col("rsi") >= technical["rsi_short_min"])
        &
        (pl.col("rsi") <= technical["rsi_short_max"])
        &
        (pl.col("bb_position") >=
         (1 - technical["bollinger_distance"]))
        &
        (pl.col("atr_ratio") >= strategy["min_atr_ratio"])
    )

    for timeframe in ("m1", "m5", "m15", "m30", "h1"):
        if not strategy.get(f"use_{timeframe}_confirmation", False):
            continue

        long_condition = (
            long_condition
            & (pl.col(f"{timeframe}_close") > pl.col(f"{timeframe}_sma50"))
            & (pl.col(f"{timeframe}_macd") > pl.col(f"{timeframe}_macd_signal"))
        )
        
        short_condition = (
            short_condition
            & (pl.col(f"{timeframe}_close") < pl.col(f"{timeframe}_sma50"))
            & (pl.col(f"{timeframe}_macd") < pl.col(f"{timeframe}_macd_signal"))
        )

    # long_condition = (
    #         (pl.col("ema20") > pl.col("ema50")) &
    #         (pl.col("ema50") > pl.col("ema200")) &
    #
    #         (pl.col("close") > pl.col("ema200")) &
    #
    #         (pl.col("range_position") < 0.6) &
    #
    #         (pl.col("ret_3") > 0) &
    #
    #         (pl.col("body") > 0) &
    #
    #         (pl.col("body_ratio") > 0.5) &
    #
    #         (pl.col("atr_regime") > 0.8)
    # )



    # short_condition = (
    #         (pl.col("ema20") < pl.col("ema50")) &
    #         (pl.col("ema50") < pl.col("ema200")) &
    #
    #         (pl.col("close") < pl.col("ema200")) &
    #
    #         (pl.col("range_position") > 0.4) &
    #
    #         (pl.col("ret_3") < 0) &
    #
    #         (pl.col("body") < 0) &
    #
    #         (pl.col("body_ratio") > 0.5) &
    #
    #         (pl.col("atr_regime") > 0.8)
    # )

    return df.with_columns(

        pl.when(long_condition)
        .then(1)

        .when(short_condition)
        .then(-1)

        .otherwise(0)

        .alias("signal")
    )