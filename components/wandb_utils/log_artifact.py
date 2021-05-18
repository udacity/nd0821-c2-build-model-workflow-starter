import wandb
import mlflow


def log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B, and add the artifact path to the MLFlow run
    so it can be retrieved by subsequent steps in a pipeline

    :param artifact_name: name for the artifact
    :param artifact_type: type for the artifact (just a string like "raw_data", "clean_data" and so on)
    :param artifact_description: a brief description of the artifact
    :param filename: local filename for the artifact
    :param wandb_run: current Weights & Biases run
    :return: None
    """
    # Log to W&B
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()
