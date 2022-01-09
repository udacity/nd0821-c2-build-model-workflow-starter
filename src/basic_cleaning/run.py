#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
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
    # local_path = wandb.use_artifact("sample.csv:latest").file()
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path, index_col="id" )

    ######################
    # YOUR CODE HERE     #

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    logger.info(f"Drop price outliers outside : {args.min_price}-{args.max_price}")

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("Convert last_review to datetime")

    # Save cleaned data in csv format
    tmp_artifact_path = os.path.join(args.tmp_directory, args.output_artifact)
    df.to_csv(tmp_artifact_path)
    logger.info(f"Temporary artifact saved to {tmp_artifact_path}")

    # Upload to W&B
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
    )
    # artifact.add_file("clean_sample.csv")
    artifact.add_file(tmp_artifact_path)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Cleaned dataset uploaded to wandb")
    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--tmp_directory",
        type=str,
        help="Temporary directory for dataset",
        required=True
    )

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact description",
        required=True
    )


    parser.add_argument(
        "--min_price",
        type=int,
        help="Minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Maximum price",
        required=True
    )


    args = parser.parse_args()

    go(args)
