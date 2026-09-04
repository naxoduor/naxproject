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