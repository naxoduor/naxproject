import polars as pl


def create_labels(
    df,
    horizon=30,
    threshold=0.0001
):

    future_close = (
        pl.col("close")
        .shift(-horizon)
    )

    future_return = (
        future_close /
        pl.col("close") - 1
    )

    return df.with_columns(

        future_return
        .alias("future_return")

    ).with_columns(

        pl.when(
            pl.col("future_return")
            > threshold
        )
        .then(1)

        .when(
            pl.col("future_return")
            < -threshold
        )
        .then(-1)

        .otherwise(0)

        .alias("target")
    )