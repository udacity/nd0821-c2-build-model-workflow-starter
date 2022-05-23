#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    This function defines the basic cleaning procedure
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # reading the file
    df = pd.read_csv(artifact_local_path, index_col="id")
    # Setting the minimum and maximum price
    minimum_price = args.min_price
    maximum_price = args.max_price

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

    tmp_artifact_path = os.path.join(args.tmp_directory,
                                     args.output_artifact)

    df.to_csv(tmp_artifact_path)
    logger.info(f"Artifact saved to {tmp_artifact_path}")

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(tmp_artifact_path)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Processed dataset uploaded to wandb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--tmp_directory",
        type=str,
        help="This is the directory used for the temporary storage of the dataset",
        required=True
    )

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="This is the name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="This is the name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="This defines the data type of the artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="This gives a brief description of the output",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The value of the minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The value of the maximum price",
        required=True
    )

    args = parser.parse_args()

    go(args)
