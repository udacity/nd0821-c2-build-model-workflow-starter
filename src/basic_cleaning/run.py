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
    
    logger.info("Initialising wandb")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f'Getting data')
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info('Cleaning data')
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info(f'Saving output')
    df.to_csv(args.output_artifact, index=False)

    logger.info(f'Sending result to wandb')
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    logger.info(f'Terminate wandb run')
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='Path to the CSV file to be cleaned',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='Path to the CSV cleaned file',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='Artifact type for wandb',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='Arfifact description for wandb',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='Minimum property prices considered',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='Maximum property prices considered',
        required=True
    )


    args = parser.parse_args()

    go(args)
