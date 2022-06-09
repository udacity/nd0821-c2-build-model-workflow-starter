#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="nyc_airbnb", group="basic_cleaning", job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info(f"Download artifact: {args.input_artifact}")
    local_path = wandb.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)
    
    # Drop outliers
    min_price = 10
    max_price = 350
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    cleaned_filename = "clean_sample.csv"
    df.to_csv(cleaned_filename, index=False)
    
    artifact = wandb.Artifact(
    args.output_artifact,
    type=args.output_type,
    description=args.output_description
    )
    
    artifact.add_file(args.output_artifact)
    logger.info(f"Uploading artifact: {args.output_artifact}")
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data cleansing.")


    parser.add_argument(
        "--input_artifact", 
        ## INSERT TYPE HERE: str, float or int,
        type = str,
        ## INSERT DESCRIPTION HERE,
        help = "input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        ## INSERT TYPE HERE: str, float or int,
        type = str,
        ## INSERT DESCRIPTION HERE,
        help = "The output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        ## INSERT TYPE HERE: str, float or int,
        type = str,
        ## INSERT DESCRIPTION HERE,
        help = "output type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        ## INSERT TYPE HERE: str, float or int,
        type = str,
        ## INSERT DESCRIPTION HERE,
        help = "output description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        ## INSERT TYPE HERE: str, float or int,
        type = int,
        ## INSERT DESCRIPTION HERE,
        help = "minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        ## INSERT TYPE HERE: str, float or int,
        type = int,
        ## INSERT DESCRIPTION HERE,
        help = "maximum price",
        required=True
    )


    args = parser.parse_args()

    go(args)
