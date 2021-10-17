#!/usr/bin/env python
"""
Download from Weights & Biases the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact in W&B
"""
import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.Namespace):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Download input artifact %s", args.input_artifact)
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)
    min_price = args.min_price
    max_price = args.max_price
    logger.info("Remove price outlier. Use range min: %.2f  -- max: %.2f", min_price, max_price)
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    df.to_csv(args.output_artifact, index=False)

    # upload to W&B
    logger.info("Upload output artifact %s of type %s", args.output_artifact, args.output_type)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    os.remove(args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

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

    arguments = parser.parse_args()

    go(arguments)
