# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
#
This repository is part of the **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity to predict credit card customers that are most likely to churn via machine learning.

This is a Python package that follows coding (PEP8) and engineering best practices. 
The package will also have the flexibility of being run interactively ("churn_notebook.ipynb") or from the command-line interface (CLI) ("churn_library.py")

## Files and data description
#
* data
    - bank_data.csv
* images
    - eda
        - Churn_distribution.png 
        - Customer_Age_distribution.png
        - Heatmap.png
        - Marital_Status_distribution.png
        - Total_Transc_Ct_distribution.png
    - results
        - classification_report_lr.png
        - classification_report_rf.png
        - feature_importance.png
        - roc_curve_result.png
* logs
    - churn_library.log
* models
    - logistic_model.pkl
    - rfc_model.pkl
* churn_library.py
* churn_notebook.ipynb
* churn_script_logging_and_tests.py
* Guide.ipynb
* README.md
* requirements_py3.6.txt
* requirements_py3.8.txt 

### Requirements

Running `churn_library` requires:

* scikit-learn==0.24.1
* shap==0.40.0
* joblib==1.0.1
* pandas==1.2.4
* numpy==1.20.1
* matplotlib==3.3.4
* seaborn==0.11.2
* pylint==2.7.4
* autopep8==1.5.6

## Installation
#
To install requirements:
```
python -m pip install -r requirements_py3.8.txt
```
## Running Files
#
To run the main file :
```
python churn_library.py
```
Each run result is saved in `Images` and `models` folders 

In order to test the script please run the following commands:
```
python churn_script_logging_and_tests.py
```
Result of tests are available in `logs` folder

## Test pep8 styling
#
```
pylint churn_library.py
```

```
pylint churn_script_logging_and_tests.py
```