import polars as pl

INPUT = "data/raw/EURUSD_H4.csv"
OUTPUT = "data/parquet/EURUSD_H4.parquet"


def main():

    df = (
        pl.scan_csv(INPUT,separator="\t",
        truncate_ragged_lines=True)
        .with_columns(
            pl.col("timestamp")
            .str.strptime(
                pl.Datetime,
                format="%Y-%m-%d %H:%M:%S",
                strict=False
            )
        )
        .select([
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ])
        .sort("timestamp")
        .collect()
    )

    df.write_parquet(
        OUTPUT,
        compression="zstd"
    )

    print(f"Rows: {df.height}")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()