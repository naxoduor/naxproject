import argparse

import polars as pl

INPUT = "data/raw/EURUSD_M5.csv"
OUTPUT = "data/parquet/EURUSD_M5.parquet"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    df = (
        pl.scan_csv(args.input,separator="\t",
        truncate_ragged_lines=True)
        .rename({
            "Time": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
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
        args.output,
        compression="zstd"
    )

    print(f"Rows: {df.height}")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()