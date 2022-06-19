"""
churn_library.py prepares the data/bank_data.csv file,
performs eda and trains between 2 models as a mean to
predict the customers' churn rate from the bank

Author: Gerard Sho
Creation Date:19/06/2022

"""
# library doc string
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# import libraries
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.read_csv(pth)
    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return dataframe


def perform_eda(dataframe):
    """
    perform eda on dataframe and save figures to images folder
    input:
        dataframe: pandas dataframe
    output:
        None
    """

    for col in ["Churn", "Customer_Age"]:
        plt.figure(figsize=(20, 10))
        dataframe[col].hist()
        plt.savefig(f"./images/eda/{col}_distribution")
        plt.clf()
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig(r"./images/eda/Marital_Status_distribution")
    plt.clf()
    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(r"./images/eda/Total_Trans_Ct_distribution")
    plt.clf()
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(r"./images/eda/Heatmap")
    plt.clf()

def encoder_helper(dataframe, category_lst, response=[]):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be \
                      used for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    """
    count = 0
    for cat in category_lst:
        lst = []
        lst_name = []
        group = dataframe.groupby(cat).mean()["Churn"]
        for val in dataframe[cat]:
            lst.append(group.loc[val])
        try:
            col_name = response[count]
        except (ValueError,IndexError):
            col_name = cat + "_Churn"
            lst_name.append(col_name)
        dataframe[col_name] = lst
        count = count + 1

    return dataframe


def perform_feature_engineering(dataframe, response=[]):
    """
    input:
            dataframe: pandas dataframe
            response: string of response name [optional argument that could be used for\
                      naming variables or index y column]

    output:
            xtrain: X training data
            xtest: X testing data
            ytrain: y training data
            ytest: y testing data
    """

    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    keep_cols.extend(response)
    # X = pd.DataFrame()
    xdata = dataframe[keep_cols]
    ydata = dataframe["Churn"]
    xtrain, xtest, ytrain, ytest = train_test_split(
        xdata, ydata, test_size=0.3, random_state=42
    )
    return xtrain, xtest, ytrain, ytest


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    """

    def plt_function(text1, text2, test, test_pred, train, train_pred, model_name):
        plt.rc("figure", figsize=(5, 5))
        plt.text(0.01, 1.25, str(text1), {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01,
            0.05,
            str(classification_report(test, test_pred)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(0.01, 0.6, str(text2), {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01,
            0.7,
            str(classification_report(train, train_pred)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.axis("off")
        plt.savefig(f"./images/results/classification_report_{model_name}")
        plt.clf()     

    plt_function(
        "Random Forest Train",
        "Random Forest Test",
        y_test,
        y_test_preds_rf,
        y_train,
        y_train_preds_rf,
        "rf",
    )
    plt_function(
        "Logistic Regression Train",
        "Logistic Regression Test",
        y_test,
        y_test_preds_lr,
        y_train,
        y_train_preds_lr,
        "lr",
    )


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    """
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(f"./images/{output_pth}/feature_importance.png")
    plt.clf()

def roc_plot(lrc_model, rf_model, x_test, y_test):
    """
    creates and stores roc curve in .images/results folder
    input:
        lrc_model: logistic regression model
        lrc_model: random forest model
        x_test: training response values
        y_test: test response values
    output:
        None
    """
    lrc_plot = plot_roc_curve(lrc_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        rf_model.best_estimator_, x_test, y_test, ax=ax, alpha=0.8
    )
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(r"./images/results/roc_curve_result.png")
    plt.clf()

def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
          x_train: X training data
          x_test: X testing data
          y_train: y training data
          y_test: y testing data
    output:
          None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    roc_plot(lrc, cv_rfc, x_test, y_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
    feature_importance_plot(cv_rfc, x_train, "results")
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")


if __name__ == "__main__":
    df = import_data("data/bank_data.csv")
    perform_eda(df)
    encoded_df = encoder_helper(
        df,
        [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ],
    )
    xtrain_, xtest_, ytrain_, ytest_ = perform_feature_engineering(encoded_df)

    train_models(xtrain_, xtest_, ytrain_, ytest_)
