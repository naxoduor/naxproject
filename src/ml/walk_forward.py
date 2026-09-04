import numpy as np

from src.ml.model import (
    create_model,
    FEATURES
)


def walk_forward(
    df,
    config
):

    train_size = config["ml"]["train_size"]
    test_size = config["ml"]["test_size"]

    X = df.select(FEATURES).to_numpy()

    y = df["target"].to_numpy()

    timestamps = df["timestamp"].to_list()

    close = df["close"].to_numpy()

    atr = df["atr"].to_numpy()

    results = []

    start = 0

    print(start, train_size, test_size, len(df))

    while (
        start + train_size + test_size
        <= len(df)
    ):

        train_end = (
            start + train_size
        )

        test_end = (
            train_end + test_size
        )

        X_train = X[
            start:train_end
        ]

        y_train = y[
            start:train_end
        ]

        X_test = X[
            train_end:test_end
        ]

        y_test = y[
            train_end:test_end
        ]


        model = create_model(
            config
        )

        model.fit(
            X_train,
            y_train
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        classes = model.classes_

        for i, probs in enumerate(
            probabilities
        ):

            row = {
                "index":
                    train_end + i,

                "timestamp":
                    timestamps[
                        train_end + i
                    ],

                "close":
                    close[
                        train_end + i
                    ],

                "atr":
                    atr[
                        train_end + i
                    ],

                "actual":
                    y_test[i],

                "probability_long": 0,

                "probability_short": 0
            }

            for cls, probability in zip(
                classes,
                probs
            ):

                if cls == 1:

                    row[
                        "probability_long"
                    ] = probability

                elif cls == -1:

                    row[
                        "probability_short"
                    ] = probability

            results.append(row)

        print(
            f"Train: {start:,}-"
            f"{train_end:,} | "
            f"Test: {train_end:,}-"
            f"{test_end:,}"
        )

        start += test_size

    return results