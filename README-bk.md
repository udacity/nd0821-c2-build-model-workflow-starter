# Build an ML Pipeline for Short-term Rental Prices in NYC

## Dev Environment

MacbookAir M1 (late 2020)

- python=3.9.13

### get_data

Downloads a sample data set from github and uploads it as an artifact into wandb. The
path and file name is obtained from the top level `config.yaml` file.
```
mlflow run . -P steps=download
```

### data_cleaning

Input: artifact from above

```
mlflow run . -P steps=basic_cleaning
```
