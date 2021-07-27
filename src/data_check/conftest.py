"""
Fixtures and configs for data_check step
"""
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    """
    Parameter parser
    """
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session', name='data')
def data_(request):
    """
    Get dataframe from csv provided
    """
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    dataframe = pd.read_csv(data_path)

    return dataframe


@pytest.fixture(scope='session', name='ref_data')
def ref_data_(request):
    """
    Get reference dataframe from csv provided
    """
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    dataframe = pd.read_csv(data_path)

    return dataframe


@pytest.fixture(scope='session', name='kl_threshold')
def kl_threshold_(request):
    """
    KL threshold parameter manager
    """
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session', name='min_price')
def min_price_(request):
    """
    Request parameter manager
    """
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session', name='max_price')
def max_price_(request):
    """
    Maximum price parameter manager
    """
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
