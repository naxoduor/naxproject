import duckdb
import polars as pl


def load_data(path):

    query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM read_parquet('{path}')
        ORDER BY timestamp
    """

    return duckdb.sql(query).pl()


def join_timeframe_features(
    base,
    lower,
    timeframe,
    timeframe_minutes,
    base_timeframe_minutes,
):

    available_at = f"{timeframe}_available_at"
    decision_time = f"{timeframe}_decision_time"

    lower_features = (
        lower
        .with_columns(
            (pl.col("timestamp") + pl.duration(minutes=timeframe_minutes))
            .alias(available_at)
        )
        .select(
            available_at,
            pl.col("close").alias(f"{timeframe}_close"),
            pl.col("sma50").alias(f"{timeframe}_sma50"),
            pl.col("macd").alias(f"{timeframe}_macd"),
            pl.col("macd_signal").alias(f"{timeframe}_macd_signal"),
        )
        .drop_nulls()
        .sort(available_at)
    )

    base = (
        base
        .with_columns(
            (pl.col("timestamp") + pl.duration(minutes=base_timeframe_minutes))
            .alias(decision_time)
        )
        .sort(decision_time)
    )

    return (
        base
        .join_asof(
            lower_features,
            left_on=decision_time,
            right_on=available_at,
            strategy="backward",
            tolerance="1h",
        )
        .drop(decision_time, available_at)
    )


def load_range(path, start=None, end=None):

    conditions = []

    if start:
        conditions.append(
            f"timestamp >= '{start}'"
        )

    if end:
        conditions.append(
            f"timestamp < '{end}'"
        )

    where = ""

    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM read_parquet('{path}')
        {where}
        ORDER BY timestamp
    """

    return duckdb.sql(query).pl()