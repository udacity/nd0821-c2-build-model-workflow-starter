# Build an ML Pipeline for Short-term Rental Prices in NYC

## Dev Environment

- Macbook x86_64 (late 2020), Monterey 12.4
- python=3.9.13
- Follow README to create a compatible Python/Conda environment

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

#### 3) Data Checks

- Input: Sample reference artifact from wandb.
- Input: Sample data (snapshot) artifact from wandb.
- Output: Logs `pytest` results to standard output.

```
mlflow run . -P steps="data_check"
```

#### 4) Initial Training

- Split data into test and training sets.
  - Input: Sample data artifact from wandb.
  - Output: Training artifact to wandb.
  - Output: Test artifact to wandb.

```
mlflow run . -P steps="data_split"
```

- Train Model
- Hyper Parameter Tuning
- Save Model

#### 4) Pipeline Release and Updates

#### Run Entire Pipeline
```
mlflow run . -P steps="all"
```

### Misc Commands
```
wandb artifact ls nyc_airbnb

wandb artifact get nyc_airbnb/clean_sample.csv:v0
```
