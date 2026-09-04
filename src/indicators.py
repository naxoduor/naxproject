import numpy as np
import polars as pl


def add_indicators(df):

    close = pl.col("close")
    high = pl.col("high")
    low = pl.col("low")

    open_ = pl.col("open")
    volume = pl.col("volume")

    df = df.with_columns([

        close.pct_change().alias("returns"),

        close.rolling_mean(10).alias("sma10"),

        close.rolling_mean(20).alias("sma20"),

        close.rolling_mean(50).alias("sma50"),

        close.rolling_mean(100).alias("sma100"),

        close.rolling_mean(200).alias("sma200"),

        close.rolling_std(20).alias("rolling_std"),

    ])

    # True range
    df = df.with_columns(

        pl.max_horizontal([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ]).alias("tr")
    )

    # ATR
    df = df.with_columns(
        pl.col("tr")
        .rolling_mean(14)
        .alias("atr")
    )

    # ATR ratio
    df = df.with_columns(
        (pl.col("atr") / close)
        .alias("atr_ratio")
    )

    # RSI
    delta = close.diff()

    df = df.with_columns([

        delta.clip(lower_bound=0)
        .rolling_mean(14)
        .alias("gain"),

        (-delta.clip(upper_bound=0))
        .rolling_mean(14)
        .alias("loss")
    ])

    df = df.with_columns(

        (
            100
            - (
                100 /
                (
                    1 +
                    pl.col("gain") /
                    pl.col("loss")
                )
            )
        ).alias("rsi")
    )

    # Bollinger Bands
    df = df.with_columns([

        (
            close.rolling_mean(20)
            + 2 * close.rolling_std(20)
        ).alias("bb_upper"),

        (
            close.rolling_mean(20)
            - 2 * close.rolling_std(20)
        ).alias("bb_lower"),

        close.rolling_mean(20)
        .alias("bb_middle")
    ])

    df = df.with_columns(

        (
            (close - pl.col("bb_lower")) /
            (
                pl.col("bb_upper")
                - pl.col("bb_lower")
            )
        ).alias("bb_position")
    )

    # MACD
    df = df.with_columns([

        close.ewm_mean(
            span=12,
            adjust=False
        ).alias("ema12"),

        close.ewm_mean(
            span=26,
            adjust=False
        ).alias("ema26")
    ])

    df = df.with_columns(

        (
            pl.col("ema12") -
            pl.col("ema26")
        ).alias("macd")
    )

    df = df.with_columns(

        pl.col("macd")
        .ewm_mean(
            span=9,
            adjust=False
        )
        .alias("macd_signal")
    )

    df = df.with_columns(

        (
            pl.col("macd") -
            pl.col("macd_signal")
        ).alias("macd_hist")
    )

    # Stochastic
    df = df.with_columns([

        low
        .rolling_min(14)
        .alias("lowest_low"),

        high
        .rolling_max(14)
        .alias("highest_high")
    ])

    df = df.with_columns(

        (
            100 *
            (
                (close - pl.col("lowest_low")) /
                (
                    pl.col("highest_high")
                    - pl.col("lowest_low")
                )
            )
        ).alias("stoch_k")
    )

    df = df.with_columns(

        pl.col("stoch_k")
        .rolling_mean(3)
        .alias("stoch_d")
    )

    # Trend relationships
    df = df.with_columns([

        (
            pl.col("sma10") /
            pl.col("sma50")
        ).alias("sma10_sma50"),

        (
            pl.col("sma50") /
            pl.col("sma100")
        ).alias("sma50_sma100"),

        (
            pl.col("sma50") -
            pl.col("sma100")
        ).alias("sma50_minus_sma100"),

    ])
    #=====================================
    df = df.with_columns([

            (
                high - low
            ).alias("candle_range"),

            (
                close - open_
            ).alias("candle_body"),

            (
                (close - open_) /
                (high - low)
            ).alias("body_ratio"),

            (
                (high - pl.max_horizontal([open_, close])) /
                (high - low)
            ).alias("upper_wick_ratio"),

            (
                (pl.min_horizontal([open_, close]) - low) /
                (high - low)
            ).alias("lower_wick_ratio"),

        ])
    #

    # ============================================================
    #     # CCI
    #     # ============================================================
    #

    typical_price = (
        high + low + close
    ) / 3
    cci_mean = typical_price.rolling_mean(20)
    mean_deviation = (
        typical_price - cci_mean
    ).abs().rolling_mean(20)
    df = df.with_columns(
        (
            (
                typical_price -
                cci_mean
            ) /
            (
                0.015 *
                mean_deviation
            )
        ).alias("cci")
    )
    #
    #     # ============================================================
    #     # WILLIAMS %R
    #     # ============================================================
    #
    df = df.with_columns(

        (
            -100 *
            (
                pl.col("highest_high") -
                close
            ) /
            (
                pl.col("highest_high") -
                pl.col("lowest_low")
            )
        ).alias("williams_r")
    )
    #
    #     # ============================================================
    #     # DONCHIAN CHANNEL
    #     # ============================================================
    #
    df = df.with_columns([

        high
        .rolling_max(20)
        .alias("donchian_high"),

        low
        .rolling_min(20)
        .alias("donchian_low"),

    ])
    #
    df = df.with_columns([

        (
            (
                close -
                pl.col("donchian_low")
            ) /
            (
                pl.col("donchian_high") -
                pl.col("donchian_low")
            )
        ).alias("donchian_position"),

        (
            close >
            pl.col("donchian_high").shift(1)
        ).alias("donchian_breakout_up"),

        (
            close <
            pl.col("donchian_low").shift(1)
        ).alias("donchian_breakout_down"),

    ])
    #
    #     # ============================================================
    #     # ADX / DIRECTIONAL MOVEMENT
    #     # ============================================================
    #
    plus_dm = (
        high.diff()
    )

    minus_dm = (
        -low.diff()
    )

    df = df.with_columns([

        pl.when(
            (plus_dm > minus_dm) &
            (plus_dm > 0)
        )
        .then(plus_dm)
        .otherwise(0)
        .alias("plus_dm"),

        pl.when(
            (minus_dm > plus_dm) &
            (minus_dm > 0)
        )
        .then(minus_dm)
        .otherwise(0)
        .alias("minus_dm"),

    ])
    #
    df = df.with_columns([

        (
            100 *
            pl.col("plus_dm")
            .rolling_mean(14) /
            pl.col("atr")
        ).alias("plus_di"),

        (
            100 *
            pl.col("minus_dm")
            .rolling_mean(14) /
            pl.col("atr")
        ).alias("minus_di"),

    ])
    
    df = df.with_columns(

        (
            100 *
            (
                (
                    pl.col("plus_di") -
                    pl.col("minus_di")
                ).abs()
            ) /
            (
                pl.col("plus_di") +
                pl.col("minus_di")
            )
        ).alias("dx")
    )

    df = df.with_columns(

        pl.col("dx")
        .rolling_mean(14)
        .alias("adx")
    )

    return df


# def add_indicators(df):
#
#     close = pl.col("close")
#     high = pl.col("high")
#     low = pl.col("low")
#     open_ = pl.col("open")
#     volume = pl.col("volume")
#
#     # ============================================================
#     # BASIC RETURNS
#     # ============================================================
#
#     df = df.with_columns([
#
#         close.pct_change().alias("returns"),
#
#         close.pct_change(3).alias("momentum_3"),
#
#         close.pct_change(5).alias("momentum_5"),
#
#         close.pct_change(10).alias("momentum_10"),
#
#         close.pct_change(20).alias("momentum_20"),
#
#     ])
#
#     # ============================================================
#     # MOVING AVERAGES
#     # ============================================================
#
#     df = df.with_columns([
#
#         close.rolling_mean(10).alias("sma10"),
#
#         close.rolling_mean(20).alias("sma20"),
#
#         close.rolling_mean(50).alias("sma50"),
#
#         close.rolling_mean(100).alias("sma100"),
#
#         close.rolling_mean(200).alias("sma200"),
#
#         close.ewm_mean(
#             span=9,
#             adjust=False
#         ).alias("ema9"),
#
#         close.ewm_mean(
#             span=21,
#             adjust=False
#         ).alias("ema21"),
#
#         close.ewm_mean(
#             span=50,
#             adjust=False
#         ).alias("ema50"),
#
#         close.ewm_mean(
#             span=100,
#             adjust=False
#         ).alias("ema100"),
#
#     ])
#
#     # ============================================================
#     # MOVING AVERAGE RELATIONSHIPS
#     # ============================================================
#
#     df = df.with_columns([
#
#         (
#             pl.col("ema9") /
#             pl.col("ema21")
#         ).alias("ema9_ema21"),
#
#         (
#             pl.col("ema21") /
#             pl.col("ema50")
#         ).alias("ema21_ema50"),
#
#         (
#             pl.col("ema50") /
#             pl.col("ema100")
#         ).alias("ema50_ema100"),
#
#         (
#             pl.col("close") /
#             pl.col("ema21") - 1
#         ).alias("price_vs_ema21"),
#
#         (
#             pl.col("close") /
#             pl.col("ema50") - 1
#         ).alias("price_vs_ema50"),
#
#         (
#             pl.col("close") /
#             pl.col("sma200") - 1
#         ).alias("price_vs_sma200"),
#
#         (
#             pl.col("sma50") /
#             pl.col("sma100")
#         ).alias("sma50_sma100"),
#
#         (
#             pl.col("sma10") /
#             pl.col("sma50")
#         ).alias("sma10_sma50"),
#
#         (
#             pl.col("sma50") -
#             pl.col("sma100")
#         ).alias("sma50_minus_sma100"),
#
#     ])
#
#     # ============================================================
#     # TRUE RANGE / ATR
#     # ============================================================
#
#     df = df.with_columns(
#
#         pl.max_horizontal([
#             high - low,
#             (high - close.shift(1)).abs(),
#             (low - close.shift(1)).abs()
#         ]).alias("tr")
#     )
#
#     df = df.with_columns([
#
#         pl.col("tr")
#         .rolling_mean(14)
#         .alias("atr"),
#
#         pl.col("tr")
#         .rolling_mean(7)
#         .alias("atr7"),
#
#         pl.col("tr")
#         .rolling_mean(28)
#         .alias("atr28"),
#
#     ])
#
#     df = df.with_columns([
#
#         (
#             pl.col("atr") / close
#         ).alias("atr_ratio"),
#
#         (
#             pl.col("atr7") /
#             pl.col("atr28")
#         ).alias("atr_short_long"),
#
#     ])
#
#     # ============================================================
#     # RSI
#     # ============================================================
#
#     delta = close.diff()
#
#     df = df.with_columns([
#
#         delta.clip(lower_bound=0)
#         .rolling_mean(14)
#         .alias("gain"),
#
#         (-delta.clip(upper_bound=0))
#         .rolling_mean(14)
#         .alias("loss")
#
#     ])
#
#     df = df.with_columns(
#
#         (
#             100 -
#             (
#                 100 /
#                 (
#                     1 +
#                     pl.col("gain") /
#                     pl.col("loss")
#                 )
#             )
#         ).alias("rsi")
#     )
#
#     # RSI momentum
#
#     df = df.with_columns([
#
#         pl.col("rsi")
#         .diff()
#         .alias("rsi_change"),
#
#         pl.col("rsi")
#         .rolling_mean(5)
#         .alias("rsi_ma"),
#
#     ])
#
#     # ============================================================
#     # BOLLINGER BANDS
#     # ============================================================
#
#     bb_middle = close.rolling_mean(20)
#     bb_std = close.rolling_std(20)
#
#     df = df.with_columns([
#
#         (
#             bb_middle +
#             2 * bb_std
#         ).alias("bb_upper"),
#
#         (
#             bb_middle -
#             2 * bb_std
#         ).alias("bb_lower"),
#
#         bb_middle.alias("bb_middle"),
#
#     ])
#
#     df = df.with_columns([
#
#         (
#             (close - pl.col("bb_lower")) /
#             (
#                 pl.col("bb_upper") -
#                 pl.col("bb_lower")
#             )
#         ).alias("bb_position"),
#
#         (
#             (
#                 pl.col("bb_upper") -
#                 pl.col("bb_lower")
#             ) /
#             pl.col("bb_middle")
#         ).alias("bb_width"),
#
#     ])
#
#     # ============================================================
#     # MACD
#     # ============================================================
#
#     df = df.with_columns([
#
#         close.ewm_mean(
#             span=12,
#             adjust=False
#         ).alias("ema12"),
#
#         close.ewm_mean(
#             span=26,
#             adjust=False
#         ).alias("ema26"),
#
#     ])
#
#     df = df.with_columns(
#
#         (
#             pl.col("ema12") -
#             pl.col("ema26")
#         ).alias("macd")
#     )
#
#     df = df.with_columns(
#
#         pl.col("macd")
#         .ewm_mean(
#             span=9,
#             adjust=False
#         )
#         .alias("macd_signal")
#     )
#
#     df = df.with_columns([
#
#         (
#             pl.col("macd") -
#             pl.col("macd_signal")
#         ).alias("macd_hist"),
#
#         pl.col("macd_hist")
#         .diff()
#         .alias("macd_hist_change"),
#
#     ])
#
#     # ============================================================
#     # STOCHASTIC
#     # ============================================================
#
#     df = df.with_columns([
#
#         low
#         .rolling_min(14)
#         .alias("lowest_low"),
#
#         high
#         .rolling_max(14)
#         .alias("highest_high"),
#
#     ])
#
#     df = df.with_columns(
#
#         (
#             100 *
#             (
#                 (close - pl.col("lowest_low")) /
#                 (
#                     pl.col("highest_high") -
#                     pl.col("lowest_low")
#                 )
#             )
#         ).alias("stoch_k")
#     )
#
#     df = df.with_columns(
#
#         pl.col("stoch_k")
#         .rolling_mean(3)
#         .alias("stoch_d")
#     )
#
#     df = df.with_columns([
#
#         (
#             pl.col("stoch_k") -
#             pl.col("stoch_d")
#         ).alias("stoch_diff"),
#
#         pl.col("stoch_k")
#         .diff()
#         .alias("stoch_change"),
#
#     ])
#
#     # ============================================================
#     # ROC
#     # ============================================================
#
#     df = df.with_columns([
#
#         (
#             (
#                 close /
#                 close.shift(5)
#             ) - 1
#         ).alias("roc5"),
#
#         (
#             (
#                 close /
#                 close.shift(10)
#             ) - 1
#         ).alias("roc10"),
#
#         (
#             (
#                 close /
#                 close.shift(20)
#             ) - 1
#         ).alias("roc20"),
#
#     ])
#
#     # ============================================================
#     # CCI
#     # ============================================================
#
#     typical_price = (
#         high + low + close
#     ) / 3
#
#     cci_mean = typical_price.rolling_mean(20)
#
#     mean_deviation = (
#         typical_price - cci_mean
#     ).abs().rolling_mean(20)
#
#     df = df.with_columns(
#
#         (
#             (
#                 typical_price -
#                 cci_mean
#             ) /
#             (
#                 0.015 *
#                 mean_deviation
#             )
#         ).alias("cci")
#     )
#
#     # ============================================================
#     # WILLIAMS %R
#     # ============================================================
#
#     df = df.with_columns(
#
#         (
#             -100 *
#             (
#                 pl.col("highest_high") -
#                 close
#             ) /
#             (
#                 pl.col("highest_high") -
#                 pl.col("lowest_low")
#             )
#         ).alias("williams_r")
#     )
#
#     # ============================================================
#     # DONCHIAN CHANNEL
#     # ============================================================
#
#     df = df.with_columns([
#
#         high
#         .rolling_max(20)
#         .alias("donchian_high"),
#
#         low
#         .rolling_min(20)
#         .alias("donchian_low"),
#
#     ])
#
#     df = df.with_columns([
#
#         (
#             (
#                 close -
#                 pl.col("donchian_low")
#             ) /
#             (
#                 pl.col("donchian_high") -
#                 pl.col("donchian_low")
#             )
#         ).alias("donchian_position"),
#
#         (
#             close >
#             pl.col("donchian_high").shift(1)
#         ).alias("donchian_breakout_up"),
#
#         (
#             close <
#             pl.col("donchian_low").shift(1)
#         ).alias("donchian_breakout_down"),
#
#     ])
#
#     # ============================================================
#     # ADX / DIRECTIONAL MOVEMENT
#     # ============================================================
#
#     plus_dm = (
#         high.diff()
#     )
#
#     minus_dm = (
#         -low.diff()
#     )
#
#     df = df.with_columns([
#
#         pl.when(
#             (plus_dm > minus_dm) &
#             (plus_dm > 0)
#         )
#         .then(plus_dm)
#         .otherwise(0)
#         .alias("plus_dm"),
#
#         pl.when(
#             (minus_dm > plus_dm) &
#             (minus_dm > 0)
#         )
#         .then(minus_dm)
#         .otherwise(0)
#         .alias("minus_dm"),
#
#     ])
#
#     df = df.with_columns([
#
#         (
#             100 *
#             pl.col("plus_dm")
#             .rolling_mean(14) /
#             pl.col("atr")
#         ).alias("plus_di"),
#
#         (
#             100 *
#             pl.col("minus_dm")
#             .rolling_mean(14) /
#             pl.col("atr")
#         ).alias("minus_di"),
#
#     ])
#
#     df = df.with_columns(
#
#         (
#             100 *
#             (
#                 (
#                     pl.col("plus_di") -
#                     pl.col("minus_di")
#                 ).abs()
#             ) /
#             (
#                 pl.col("plus_di") +
#                 pl.col("minus_di")
#             )
#         ).alias("dx")
#     )
#
#     df = df.with_columns(
#
#         pl.col("dx")
#         .rolling_mean(14)
#         .alias("adx")
#     )
#
#     # ============================================================
#     # PRICE ACTION / CANDLE STRUCTURE
#     # ============================================================
#
#     df = df.with_columns([
#
#         (
#             high - low
#         ).alias("candle_range"),
#
#         (
#             close - open_
#         ).alias("candle_body"),
#
#         (
#             (close - open_) /
#             (high - low)
#         ).alias("body_ratio"),
#
#         (
#             (high - pl.max_horizontal([open_, close])) /
#             (high - low)
#         ).alias("upper_wick_ratio"),
#
#         (
#             (pl.min_horizontal([open_, close]) - low) /
#             (high - low)
#         ).alias("lower_wick_ratio"),
#
#     ])
#
#     # ============================================================
#     # PRICE POSITION IN RECENT RANGE
#     # ============================================================
#
#     df = df.with_columns([
#
#         high
#         .rolling_max(50)
#         .alias("recent_high_50"),
#
#         low
#         .rolling_min(50)
#         .alias("recent_low_50"),
#
#         high
#         .rolling_max(100)
#         .alias("recent_high_100"),
#
#         low
#         .rolling_min(100)
#         .alias("recent_low_100"),
#
#     ])
#
#     df = df.with_columns([
#
#         (
#             (close - pl.col("recent_low_50")) /
#             (
#                 pl.col("recent_high_50") -
#                 pl.col("recent_low_50")
#             )
#         ).alias("range_position_50"),
#
#         (
#             (close - pl.col("recent_low_100")) /
#             (
#                 pl.col("recent_high_100") -
#                 pl.col("recent_low_100")
#             )
#         ).alias("range_position_100"),
#
#     ])
#
#     # ============================================================
#     # VOLUME / ACTIVITY
#     # ============================================================
#
#     df = df.with_columns([
#
#         volume.rolling_mean(20)
#         .alias("volume_sma20"),
#
#         volume.pct_change()
#         .alias("volume_change"),
#
#     ])
#
#     df = df.with_columns(
#
#         (
#             volume /
#             pl.col("volume_sma20")
#         ).alias("volume_ratio")
#     )
#
#     return df


# def add_indicators(df):
#
#     close = pl.col("close")
#     high = pl.col("high")
#     low = pl.col("low")
#
#     # ------------------------------------------------
#     # RETURNS / MOMENTUM
#     # ------------------------------------------------
#
#     df = df.with_columns([
#         close.pct_change(1).alias("ret_1"),
#         close.pct_change(3).alias("ret_3"),
#         close.pct_change(5).alias("ret_5"),
#         close.pct_change(10).alias("ret_10"),
#     ])
#
#     # ------------------------------------------------
#     # ATR
#     # ------------------------------------------------
#
#     df = df.with_columns(
#         pl.max_horizontal([
#             high - low,
#             (high - close.shift(1)).abs(),
#             (low - close.shift(1)).abs()
#         ]).alias("tr")
#     )
#
#     df = df.with_columns(
#         pl.col("tr")
#         .rolling_mean(14)
#         .alias("atr")
#     )
#
#     # ------------------------------------------------
#     # TREND
#     # ------------------------------------------------
#
#     df = df.with_columns([
#         close.ewm_mean(span=20, adjust=False).alias("ema20"),
#         close.ewm_mean(span=50, adjust=False).alias("ema50"),
#         close.ewm_mean(span=200, adjust=False).alias("ema200"),
#     ])
#
#     df = df.with_columns([
#         (
#             (close - pl.col("ema50")) /
#             pl.col("atr")
#         ).alias("distance_ema50_atr"),
#
#         (
#             (close - pl.col("ema200")) /
#             pl.col("atr")
#         ).alias("distance_ema200_atr"),
#
#         (
#             pl.col("ema20") -
#             pl.col("ema50")
#         ).alias("ema20_50_diff"),
#
#         (
#             pl.col("ema50") -
#             pl.col("ema200")
#         ).alias("ema50_200_diff"),
#     ])
#
#     # ------------------------------------------------
#     # VOLATILITY REGIME
#     # ------------------------------------------------
#
#     df = df.with_columns([
#         (
#             pl.col("atr") /
#             close
#         ).alias("atr_pct"),
#
#         (
#             pl.col("atr") /
#             pl.col("atr").rolling_mean(50)
#         ).alias("atr_regime"),
#     ])
#
#     # ------------------------------------------------
#     # RECENT STRUCTURE
#     # ------------------------------------------------
#
#     df = df.with_columns([
#         high.shift(1)
#         .rolling_max(20)
#         .alias("recent_high"),
#
#         low.shift(1)
#         .rolling_min(20)
#         .alias("recent_low"),
#     ])
#
#     # Position inside recent range
#     df = df.with_columns(
#         (
#             (close - pl.col("recent_low")) /
#             (
#                 pl.col("recent_high") -
#                 pl.col("recent_low")
#             )
#         ).alias("range_position")
#     )
#
#     # ------------------------------------------------
#     # BREAKOUT
#     # ------------------------------------------------
#
#     df = df.with_columns([
#         (
#             close >
#             pl.col("recent_high")
#         ).cast(pl.Int8).alias("breakout_up"),
#
#         (
#             close <
#             pl.col("recent_low")
#         ).cast(pl.Int8).alias("breakout_down"),
#     ])
#
#     # ------------------------------------------------
#     # CANDLE STRUCTURE
#     # ------------------------------------------------
#
#     df = df.with_columns([
#         (high - low).alias("candle_range"),
#
#         (
#             close - pl.col("open")
#         ).alias("body"),
#
#         (
#             high -
#             pl.max_horizontal([
#                 close,
#                 pl.col("open")
#             ])
#         ).alias("upper_wick"),
#
#         (
#             pl.min_horizontal([
#                 close,
#                 pl.col("open")
#             ]) - low
#         ).alias("lower_wick"),
#     ])
#
#     df = df.with_columns([
#         (
#             pl.col("body") /
#             pl.col("candle_range")
#         ).alias("body_ratio"),
#
#         (
#             pl.col("upper_wick") /
#             pl.col("candle_range")
#         ).alias("upper_wick_ratio"),
#
#         (
#             pl.col("lower_wick") /
#             pl.col("candle_range")
#         ).alias("lower_wick_ratio"),
#     ])
#
#     return df