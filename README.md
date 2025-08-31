# PUBG Game Prediction

<center><img src="https://media.giphy.com/media/XVbrX433vn6rqkexSj/giphy.gif"></center>

## Overview

This project predicts the winning placement percentile (`winPlacePerc`) in PUBG games using machine learning. Data is wrangled, features engineered, and a CatBoost regressor trained.

- **Dataset:** PUBG_Game_Prediction_data.csv (4.4M rows, 29 columns)  
- **Key Steps:** Import libraries, read/clean data, feature engineering (e.g., normalized kills/damage, team aggregates), train CatBoost model.  
- **Performance:** RMSE = 0.08, R² = 0.93  

For full details, see [PUBG Game Prediction.ipynb](PUBG%20Game%20Prediction.ipynb).

## Table of Contents

1. [Importing Libraries](#importing-libraries)  
2. [Reading Data](#reading-data)  
3. [Data Wrangling](#data-wrangling)  
4. [Feature Engineering](#feature-engineering)  
5. [ML - CatBoost Model](#ml---catboost-model)  

## Importing Libraries

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = (16, 6)
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import catboost as cb
from sklearn.metrics import mean_squared_error, r2_score
```

Install CatBoost: `pip install catboost`

## Reading Data

```python
df = pd.read_csv("PUBG_Game_Prediction_data.csv")
df.head()  # Shows first 5 rows
df.shape   # (4446966, 29)
df.info()  # Data types and memory info
```

(See notebook for data description.)

## Data Wrangling

- Remove row with null `winPlacePerc`.  
- Add `playersJoined` for match player count.  
- Handle anomalies (e.g., cheaters with 0 walkDistance but kills).  
- Normalize features based on players in match.

(See notebook for full cleaning code.)

## Feature Engineering

- Normalized features: `killsNorm`, `damageDealtNorm`, etc.  
- Aggregates: `totalDistance` (walk + ride + swim), `playersInTeam`, `healsAndBoosts`.  
- Drop irrelevant columns like `teamKills`, `numGroups`.

## ML - CatBoost Model

```python
# Split data
x = df.drop(['winPlacePerc', 'teamKills', 'numGroups'], axis=1)  # Adjust as needed
y = df['winPlacePerc']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

# Train
model = cb.CatBoostRegressor(iterations=100, eval_metric='MAE', verbose=20)
model.fit(xtrain, ytrain, eval_set=(xtest, ytest), use_best_model=True)
```

- Feature importance plotted (e.g., walkDistance highest).  
- Predictions: **RMSE = 0.08, R² = 0.93**.

<center><img src="https://media.giphy.com/media/KB89dMAtH79VIvxNCW/giphy.gif"></center>

## Requirements

- Python 3.x  
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, catboost  

```bash
pip install -r requirements.txt
```
