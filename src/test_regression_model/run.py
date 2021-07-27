#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def execute(args):
    """
    Test regression model procedure
    """
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    x_test = pd.read_csv(test_dataset_path)
    y_test = x_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(x_test)

    logger.info("Scoring")
    r_squared = sk_pipe.score(x_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info("Score: %s", r_squared)
    logger.info("MAE: %s", mae)

    # Log MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    main_args = parser.parse_args()

    execute(main_args)
