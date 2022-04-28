#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import os

import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    data = pd.read_csv(artifact_local_path, index_col="id")
    min_price = args.min_price
    max_price = args.max_price

    idx = data['price'].between(min_price, max_price)
    data = data[idx].copy()
    logger.info("Basic cleaning: Removed prices out of [%s - %s]",
                min_price, max_price)

    data['last_review'] = pd.to_datetime(data['last_review'])
    logger.info("Basic cleaning: Fixed data type for last_review column")

    output_file = args.output_artifact
    data.to_csv(output_file)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)
    logger.info("Basic cleaning: logged %s artifact", output_file)

    os.remove(artifact_local_path)
    os.remove(output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Initial dataset",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Dataset obtained after the cleaning, uploaded to W&B",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output data type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Output data description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price limit",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price limit",
        required=True
    )


    args = parser.parse_args()

    go(args)
