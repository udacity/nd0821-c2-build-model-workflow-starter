# Build an ML Pipeline for Short-term Rental Prices in NYC

## Dev Environment

MacbookAir M1 (late 2020)

- python=3.9.13

### Pipeline Components

#### 1) get_data

- Input: Download sample data (csv file) from github.
- Output: Upload data artifact to wandb.

```
mlflow run . -P steps=download
```

#### 2) data_cleaning

- Input: Download artifact from wandb.
- Output: Upload cleaned data artifact to wandb.

```
mlflow run . -P steps=basic_cleaning
```
