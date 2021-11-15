#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Load data
    local_path = run.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)
    logger.info(f"Loaded data from {local_path} as df: {df.shape}")

    # Drop outliers
    min_price, max_price = args.min_price, args.max_price
    logger.info(f"Only keep records with price between {min_price} and {max_price}")
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Changed type for field 'last_review'")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # export as csv
    df.to_csv(args.output_artifact, index=False)

    # Load data to W&B
    logger.info("Logging artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    os.remove(args.output_artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="the name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="the type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="a description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="the minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="the maximum price to consider",
        required=True
    )
    args = parser.parse_args()

    go(args)
