"""
Main procedure for MLops pipeline
"""
import json
import tempfile
import os
import mlflow
import hydra
from omegaconf import DictConfig

_steps = [
    "download_data",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model"
]


#pylint: disable = no-value-for-parameter
@hydra.main(config_name='config')
def execute(config: DictConfig):
    """
    Main procedure for MLops pipeline
    """
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    root_path = hydra.utils.get_original_cwd()

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download_data" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "download_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    root_path,
                    "src",
                    "basic_cleaning"),
                "main",
                parameters={
                    "tmp_directory": tmp_dir,
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']},
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "train_val_test_split"),
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into
            # JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as file_p:
                json.dump(
                    dict(
                        config["modeling"]["random_forest"].items()),
                    file_p)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for
            # the train_random_forest step
            _ = mlflow.run(
                os.path.join(
                    root_path,
                    "src",
                    "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "model_export"},
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": config["test_model"]["model"],
                    "test_dataset": "test_data.csv:latest"
                }
            )


if __name__ == "__main__":
    execute()
