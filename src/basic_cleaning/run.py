#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
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
    logger.info(f'Downloading artifact {args.input_artifact}')
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Read the data from the artifact
    logger.info("Loading Dataframe")
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Drop Outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Drop outliers in longitude and latitude columns
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Convert lastreview to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save dataframe to file
    logger.info(f'Saving Dataframe {args.output_artifact}')
    df.to_csv(args.output_artifact, index=False)

    # Create an artifact instance
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    
    artifact.add_file(args.output_artifact)

    # Log the artifact
    logger.info(f'Log artifact {args.output_artifact}')
    run.log_artifact(artifact)

    # Remove file
    logger.info(f'Removing local file {args.output_artifact}')
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
        help="Minimum price for outlier range",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price for outlier range",
        required=True
    )


    args = parser.parse_args()

    go(args)
