#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Downloading input artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Dropping price column outliers. Range min: {args.min_price} -- max: {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info(f"Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    df.to_csv(args.output_artifact, index=False)

    # Upload to W&B
    logger.info("Upload output artifact %s of type %s", args.output_artifact, args.output_type)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The input artifact and its version (i.e. artifact.csv:latest)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="The output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="min price to consider for the outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="max price to consider for the outliers",
        required=True
    )

    arguments = parser.parse_args()

    go(arguments)
