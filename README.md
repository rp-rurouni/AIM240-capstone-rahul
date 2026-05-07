# AIM240-capstone-rahul

# AIM240 Capstone: Predicting Flight Delays Using Machine Learning

## Project Overview

This capstone project builds a machine learning pipeline to predict whether a U.S. domestic flight will arrive **15 minutes or more late**. The project uses historical 2015 U.S. domestic flight data and treats the problem as a binary classification task:

- `1` = flight arrived 15+ minutes late
- `0` = flight arrived less than 15 minutes late or on time

The goal is not to perfectly predict every delay, but to build a clear and reusable machine learning workflow that includes data cleaning, feature engineering, model training, baseline comparison, model evaluation, and single-flight prediction.

## Dataset

The project uses the Kaggle 2015 U.S. flight delay dataset, which contains more than 5 million flight records. The dataset includes airline, airport, schedule, distance, delay, cancellation, and timing information.

The target variable is created from `ARRIVAL_DELAY`:

```python
IS_DELAYED = 1 if ARRIVAL_DELAY >= 15
IS_DELAYED = 0 otherwise
````

Cancelled and diverted flights are removed because they do not have a meaningful arrival delay value for this classification problem.

## Main Workflow

The notebook includes the following steps:

1. Load and inspect the flight dataset
2. Create the binary delay target variable
3. Clean the data and remove leakage columns
4. Engineer features such as departure hour and weekend flag
5. Encode airline and airport information
6. Split the data into train, validation, and test sets
7. Train a Logistic Regression baseline model
8. Train and tune a LightGBM model
9. Tune the probability threshold for better delay detection
10. Evaluate final performance on the test set
11. Save the model bundle
12. Use `predict_from_inputs()` for single-flight prediction

## Models Compared

Two main models are compared:

### Logistic Regression Baseline

Logistic Regression is used as a simple baseline model. It helps show how a simpler linear model performs before using a more advanced tree-based model.

### LightGBM Final Model

LightGBM is used as the main model because it works well with large structured/tabular datasets. The model is tuned using GridSearchCV.

Final selected LightGBM parameters:

```python
learning_rate = 0.1
max_depth = -1
min_child_samples = 20
n_estimators = 200
num_leaves = 63
```

## Evaluation

The data is split into:

* 70% training
* 15% validation
* 15% test

Because most flights are on time, accuracy alone is not enough. The project also evaluates precision, recall, F1 score, and confusion matrices, especially for the delayed-flight class.

## Final Test Results

### Logistic Regression Baseline

* Accuracy: about 59%
* Delayed-flight precision: about 25%
* Delayed-flight recall: about 63%

Logistic Regression catches many delayed flights but produces many false delay warnings.

### LightGBM with Tuned Threshold

* Accuracy: about 81%
* Delayed-flight precision: about 48%
* Delayed-flight recall: about 30%

The tuned LightGBM model provides a better balance between catching delayed flights and avoiding too many false alarms.

## Final Model Choice

The final model is the tuned LightGBM model. It does not catch every delay, but it gives a more balanced result than Logistic Regression. Logistic Regression is more aggressive and creates many false alarms, while LightGBM is more practical for this capstone use case.

## Single-Flight Prediction

The notebook includes a function called:

```python
predict_from_inputs()
```

This function allows a user to enter flight details such as airline, origin airport, destination airport, scheduled departure time, scheduled flight duration, and distance. It returns a delay probability and a prediction label.

Example output:

```python
{
  "label": "On-time (<15 min arrival delay)",
  "delay_probability": 0.1829,
  "threshold": 0.323
}
```

## Saved Model Bundle

The notebook saves the trained model and required preprocessing artifacts using `joblib`. The saved bundle includes:

* Final LightGBM model
* Selected probability threshold
* Feature column order
* Airport encoding maps
* Fallback values for unseen airports

This helps ensure that future predictions use the same preprocessing steps as training.

## Limitations

This project has several limitations:

* It uses historical 2015 flight data only.
* It does not include live weather data.
* It does not include live airport congestion or air traffic control data.
* It does not predict exact delay minutes.
* The final model still misses many delayed flights.
* This is a capstone prototype, not a production airline operations system.

## Future Improvements

Future improvements could include:

* Adding weather data
* Adding newer flight data
* Adding airport congestion data
* Comparing performance by airline or airport
* Testing different thresholds for different business needs
* Turning the prediction function into a small API or web app

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* LightGBM
* matplotlib
* seaborn
* joblib
* Google Colab / Jupyter Notebook

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Make sure the flight delay dataset is available.
3. Run the notebook cells from top to bottom.
4. Review the baseline and LightGBM test results.
5. Run the `predict_from_inputs()` example to test a single-flight prediction.

## Final Summary

This project demonstrates a complete machine learning workflow for predicting whether a U.S. domestic flight will arrive 15 minutes or more late. The final LightGBM model achieves about 81% test accuracy, with about 48% precision and 30% recall for delayed flights. The model is not perfect, but it provides a clear and reproducible capstone-level pipeline for flight delay classification.

```
```
