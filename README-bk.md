# Build an ML Pipeline for Short-term Rental Prices in NYC

## Dev Environment

MacbookAir M1 (late 2020)

- python=3.9.13

### Pipeline Components

#### 1) get_data

Downloads a sample data set from github and uploads it as an artifact into wandb. The
path and file name is obtained from the top level `config.yaml` file.

- Input: Download sample data (csv file) from github.
- Ouptut: Upload data artifact to wandb.

```
mlflow run . -P steps=download
```

#### 2) data_cleaning

- Input: Download artifact from wandb.
- Ouptut: Upload cleaned data artifact to wandb.

```
mlflow run . -P steps=basic_cleaning
```
