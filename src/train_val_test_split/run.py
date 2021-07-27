"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def execute(args):
    """
    Split data for training/testing/validating purposes
    """
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    logger.info("Fetching artifact %s", args.input)
    artifact_local_path = run.use_artifact(args.input).file()

    dataframe = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        dataframe,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=dataframe[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    for dataframe, k in zip([trainval, test], ['trainval', 'test']):
        logger.info("Uploading %s_data.csv dataset", k)
        with tempfile.NamedTemporaryFile("w") as file_p:

            dataframe.to_csv(file_p.name, index=False)

            log_artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k} split of dataset",
                file_p.name,
                run,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False)

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False)

    main_args = parser.parse_args()

    execute(main_args)
