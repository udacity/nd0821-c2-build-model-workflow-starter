#!/usr/bin/env python
"""Cleaning component.

Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact

Created on: 04/02/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args: argparse.ArgumentParser):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading artifact")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=None, ## INSERT TYPE HERE: str, float or int,
        help=None,## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=None,## INSERT TYPE HERE: str, float or int,
        help=None,## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=None,## INSERT TYPE HERE: str, float or int,
        help=None,## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=None,## INSERT TYPE HERE: str, float or int,
        help=None,## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=None,## INSERT TYPE HERE: str, float or int,
        help=None,## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=None,## INSERT TYPE HERE: str, float or int,
        help=None,## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
