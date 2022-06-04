import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def basic_cleaning():
    # basic cleaning
    sample_data_path = "components/get_data/data/sample2.csv"
    df = pd.read_csv(sample_data_path, index_col="id")
    # Setting the minimum and maximum price
    minimum_price = 10
    maximum_price = 350

    # Dealing with outliers
    index = df['price'].between(minimum_price, maximum_price)
    df = df[index].copy()
    logger.info(
        f"Removing Price outliers in dataset that is outside range {minimum_price} - {maximum_price}"
    )
    df["last_review"] = pd.to_datetime(df['last_review'])
    logger.info("Last review of dataset to effect type fix")

    index = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5,
                                                                             41.2)
    df = df[index].copy()

    tmp_artifact_path = "output_artifact.csv"

    df.to_csv(tmp_artifact_path)
    logger.info(f"Artifact saved to {tmp_artifact_path}")


def train_test_val():
    df = pd.read_csv("output_artifact.csv")
    trainval, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["neighbourhood_group"],
    )

    for df, k in zip([trainval, test], ['trainval.csv', 'test.csv']):
        df.to_csv(k, index="False")


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


def get_inference_pipeline(rf_config, max_tfidf_features):
    # Let's handle the categorical features first
    # Ordinal categorical are categorical values for which the order is meaningful, for example
    # for room type: 'Entire home/apt' > 'Private room' > 'Shared room'
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in production
    # (nor during training). That is not true for neighbourhood_group
    ordinal_categorical_preproc = OrdinalEncoder()

    ######################################
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                    OneHotEncoder())
    ######################################

    # Let's impute the numerical columns to make sure we can handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    # Create random forest
    random_Forest = RandomForestRegressor()

    ######################################
    # Create the inference pipeline. The pipeline must have 2 steps: a step called "preprocessor" applying the
    # ColumnTransformer instance that we saved in the `preprocessor` variable, and a step called "random_forest"
    # with the random forest instance that we just saved in the `random_forest` variable.
    # HINT: Use the explicit Pipeline constructor so you can assign the names to the steps, do not use make_pipeline
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            "random_forest", random_Forest
        ]
    )

    return sk_pipe, processed_features


def train_model():
    trainval_local_path = "trainval.csv"
    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes the column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=X["neighbourhood_group"], random_state=42
    )

    logger.info("Preparing sklearn pipeline")

    rf_config = {
        "n_estimators": 5,
        "max_depth": 15,
        "min_samples_split": 4,
        "min_samples_leaf": 3,
    }

    sk_pipe, processed_features = get_inference_pipeline(rf_config, 5)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    ######################################
    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    ######################################
    sk_pipe.fit(X_train, y_train)
    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    model_directory = "random_forest_dir"
    if os.path.exists(model_directory):
        shutil.rmtree(model_directory)

    ######################################
    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory "random_forest_dir"
    # HINT: use mlflow.sklearn.save_model
    # YOUR CODE HERE
    ######################################
    mlflow.sklearn.save_model(
        sk_model=sk_pipe,
        path=model_directory,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:2]
    )


if __name__ == "__main__":
    basic_cleaning()
    train_test_val()
    train_model()

