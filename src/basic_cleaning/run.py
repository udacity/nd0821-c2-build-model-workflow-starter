#!/usr/bin/env python
"""Cleaning component.

Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact

Created on: 04/02/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import os
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.ArgumentParser):

    run = wandb.init(job_type="basic_cleaning")

    run.config.update(args)

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()

    logger.info(f"Reading artifact from {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    logger.info("Processing price column")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Processing last_reviw column")
    df['last_review'] = pd.to_datetime(df['last_review'])

    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(args.output_artifact)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum rental price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum rental price",
        required=True
    )


    args = parser.parse_args()

    go(args)
