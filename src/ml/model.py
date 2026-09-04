import numpy as np

from sklearn.ensemble import RandomForestClassifier


FEATURES = [

    "returns",

    "rsi",

    "macd",
    "macd_signal",
    "macd_hist",

    "atr",
    "atr_ratio",

    "bb_position",

    "stoch_k",
    "stoch_d",

    "sma10_sma50",
    "sma50_sma100",

    "sma50_minus_sma100",

    "rolling_std",

    "aoi_position",

    "distance_support",
    "distance_resistance",


    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    # "volume_ratio",
    "adx",
    "plus_di",
    "minus_di",

]




def create_model(config):

    ml = config["ml"]

    return RandomForestClassifier(

        n_estimators=
            ml["n_estimators"],

        min_samples_leaf=
            ml["min_samples_leaf"],

        random_state=
            ml["random_state"],

        n_jobs=-1,

        class_weight="balanced_subsample"
    )


def train_model(
    model,
    X,
    y
):

    model.fit(X, y)

    return model


def probabilities(
    model,
    X
):

    return model.predict_proba(X)