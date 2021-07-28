"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import os
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def execute(args):
    """
    Basic cleaning procedure
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()

    dataframe = pd.read_csv(artifact_local_path, index_col="id")
    min_price = args.min_price
    max_price = args.max_price
    idx = dataframe['price'].between(min_price, max_price)
    dataframe = dataframe[idx].copy()
    logger.info("Dataset price outliers removal outside range: %s-%s",
                 args.min_price, args.max_price)
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
    logger.info("Dataset last_review data type fix")

    tmp_artifact_path = os.path.join(args.tmp_directory, args.output_artifact)
    dataframe.to_csv(tmp_artifact_path)
    logger.info("Temporary artifact saved to %s" , tmp_artifact_path)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(tmp_artifact_path)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Cleaned dataset uploaded to wandb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--tmp_directory",
        type=str,
        help="Temporary directory for dataset storage",
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
        help="Output artifact name",
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
        help="Minimum price limit",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Maximum price limit",
        required=True
    )

    main_args = parser.parse_args()

    execute(main_args)
