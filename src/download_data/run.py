#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import argparse
import logging
import os

import wandb

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def execute(args):
    """
    Data download and upload to Weights & Biases procedure
    """
    run = wandb.init(job_type="download_file")
    run.config.update(args)

    logger.info("Returning sample %s", args.sample)
    logger.info("Uploading %s to Weights & Biases", args.artifact_name)
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        os.path.join("data", args.sample),
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download URL to a local destination")

    parser.add_argument(
        "sample",
        type=str,
        help="Name of the sample to download")

    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name for the output artifact")

    parser.add_argument(
        "artifact_type",
        type=str,
        help="Output artifact type.")

    parser.add_argument(
        "artifact_description",
        type=str,
        help="A brief description of this artifact")

    main_args = parser.parse_args()

    execute(main_args)
