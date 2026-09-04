import polars as pl


def add_aoi_features(df):

    df = df.with_columns([

        pl.col("high")
        .rolling_max(100)
        .alias("recent_high"),

        pl.col("low")
        .rolling_min(100)
        .alias("recent_low")
    ])

    df = df.with_columns([

        (
            (pl.col("close") - pl.col("recent_low")) /
            (
                pl.col("recent_high")
                - pl.col("recent_low")
            )
        ).alias("aoi_position"),

        (
            (pl.col("recent_high") - pl.col("close")) /
            pl.col("close")
        ).alias("distance_resistance"),

        (
            (pl.col("close") - pl.col("recent_low")) /
            pl.col("close")
        ).alias("distance_support")
    ])

    # Near support
    df = df.with_columns(

        (
            pl.col("distance_support") < 0.001
        ).alias("near_support")
    )

    # Near resistance
    df = df.with_columns(

        (
            pl.col("distance_resistance") < 0.001
        ).alias("near_resistance")
    )

    return df

def clean_features(df):

    return (
        df
        .drop_nulls()
        .filter(
            pl.col("atr") > 0
        )
    )