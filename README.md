# Kaggle Challenge: House Prices
## Advanced Regression Techniques
Challenge available at https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/

The objective of this challenge is to estimate the sale price of some properties based on 80 features contained in the available training and testing datasets. The results are classified according to the root mean square log error (RMSLE): the lower the better, with the perfect score being zero.

In the file that I provide in this repository, I perform data wrangling, cleaning and imputation, also creating new features that make sense in my reasoning, through feature engineering. The procedures are quite simple after all, these processes are well explained throughout the Python file.

### About these Python files
`house_prices.py` is the to-go file.

`prices.py` is the file that I work on in my PC, kind of a mess that works for me.

## Categorical / Numerical Features
NaN values ​​are treated on a case-by-case basis, with the imputation being the mode of some other feature or its median (whichever suits), for example. After the data processing, I end up with two different databases for training the models for this challenge. Some features are kept as originally supplied, others are changed from numeric to categorical.

## Feature Engeneering
Based on the datasets, I create some new variables to help train the models:
- Age of the house, renovations and garage
- Total areas (basements, patio, etc.)
- I create new categorical variables for the existence of decks, porches and pools
- I separate the information about the number of bathrooms
- So on and so forth...

## Outlier Detection
I estimate outliers with the help of some bloxplot graphs. In the script, I present the information already filtered, but you can simply change the code a little if you want and the full information is shown in the graph. To filter outliers, I use the interquartile ranges approach.

## Pre-training adjustments: OHE
To ensure that all categorical variables are used in training, I perform One Hot Encoding (or dummyization) over the categorical variables using `scikit-learn` functions, and for study purposes I separate this result into two distinct datasets: one that has been processed through the One Hot Encoding and another that did not go through the process, so that easy comparison is possible when training the models.

## Regression models used
For this study, I code and train machine learning models based on `xgboost`, `catboost`, `lightgbm` and `randomforest` (this one from `sklearn`).

In my first approach to this problem, I used the `sklearn` GBR (gradient booster) algorithm, among others. This trained model resulted in a final RMSLE score of 0.14569, ranking me in position 2227 (top 8% at the time) on the world leaderboard at the time of submission of the results.

When reviewing this file to share here, I ended up working on it a little better and managed to improve my score to 0.13030, which took me to position 1067 at the time - a good progress in this review.

## Testing area
Some ideas of what you can change in the structure as a whole to try to improve your score:
- Normalization of variables
- Different treatment of outliers
- Different imputation methods
- Different interpretation of my reasoning presented

Did you get a better score than me, based on my file? Send me a message because I'm curious to know what we did differently!
