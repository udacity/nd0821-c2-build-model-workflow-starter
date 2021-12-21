#!/usr/bin/env python
"""
[An example of a step using MLflow and Weights & Biases]: 
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Inicialize W&B job
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info('Input artifact fetched')

    # Read the input csv artifact
    df = pd.read_csv(artifact_local_path)

    # Filter outliers in 'price' column
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review column type from str to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info('Completed cleaning processing')

    # Save resultant dataframe to a local file
    df.to_csv('clean_sample.csv', index=False)
    logger.info('Saved df to csv')

    output_artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    output_artifact.add_file('clean_sample.csv')

    run.log_artifact(output_artifact)
    logger.info('Artifact saved to W&B')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Name of input artifact raw data',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='Name of output artifact cleaned data',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='Cleaned data after EDA steps',
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help='minimum price to filter price column value',
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help='maximum price to filter price column value',
        required=True
    )

    args = parser.parse_args()

    go(args)


