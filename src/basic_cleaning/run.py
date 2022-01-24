#!/usr/bin/env python
"""
Download the raw dataset from W&B and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info('Downloading artifact from W&B')
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info('Reading dataset into pandas')
    df = pd.read_csv(artifact_path)

    logger.info('Dropping price outliers')
    min_price = 10
    max_price = 350
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    logger.info('Convert last_review column to datetime')
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info('Saving cleaned dataset')
    df.to_csv('clean_sample.csv', index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    logger.info('Loading cleaned dataset into W&B')
    artifact.add_file('clean_sample.csv')

    logger.info('Logging artifact')
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact to process",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='Output artifact filename',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='Type of output artifact to create',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='Output artifact description',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='Minimum rental price',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='Maximum rental price',
        required=True
    )


    args = parser.parse_args()

    go(args)
