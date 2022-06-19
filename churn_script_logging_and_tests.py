"""
churn_script_logging_and_tests.py tests for all the functions in churn_library.py

Author: Gerard Sho
Creation Date:19/06/2022

"""

import os
import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

def import_test():
    """
    import_test data - this example is completed for you to assist with the other test functions
    """
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("import_test : SUCCESS")
    except FileNotFoundError as err:
        logging.error("import_test : Failed - The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "import_test: The file doesn't appear to have rows and columns"
        )
        raise err
    return dataframe


def eda_test(import_test):
    """
    test perform eda function
    """
    cls.perform_eda(import_test)
    file_exists1 = os.path.exists("./images/eda/Churn_distribution.png")
    file_exists2 = os.path.exists("./images/eda/Customer_Age_distribution.png")
    file_exists3 = os.path.exists("./images/eda/Marital_Status_distribution.png")
    file_exists4 = os.path.exists("./images/eda/Total_Trans_Ct_distribution.png")
    file_exists5 = os.path.exists("./images/eda/Heatmap.png")
    try:
        assert file_exists1 == True
        assert file_exists2 == True
        assert file_exists3 == True
        assert file_exists4 == True
        assert file_exists5 == True
        logging.info("eda_test : SUCCESS - Process EDA completed")
    except AssertionError as err:
        logging.error("eda_test : Failed - EDA files are not found")
        raise err


def encoder_helper_test(import_test):
    """
    encoded dataframe fixture - returns the encoded dataframe on some specific column
    """

    dataframe_encoded = cls.encoder_helper(
        import_test,
        [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ],
    )
    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
        logging.info("encoder_helper_test : SUCCESS-Encoded dataframe fixture creation")
    except AssertionError as err:
        logging.error("encoder_helper_test : Failed-Encoded dataframe fixture creation fails")
        raise err
    return dataframe_encoded


def perform_feature_engineering_test(encoder_helper_test):
    """
    test perform_feature_engineering
    """
    train_and_test = cls.perform_feature_engineering(encoder_helper_test)
    X_train, X_test, y_train, y_test = train_and_test
    try:
        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_test) > 0
        assert len(y_test) > 0
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        logging.info("perform_feature_engineering_test : SUCCESS -  Feature_engineering successfully completed")
    except AssertionError as err:
        logging.error("perform_feature_engineering_test : Failed")
        raise err
    return train_and_test


def test_train_models(perform_feature_engineering_test):
    """
    test train_models
    """
    cls.train_models(
        perform_feature_engineering_test[0],
        perform_feature_engineering_test[1],
        perform_feature_engineering_test[2],
        perform_feature_engineering_test[3],
    )
    try:
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        logging.info("test_train_models :  SUCCESS - Models were trained")
    except AssertionError as err:
        logging.error("test_train_models :  Failed - No models can be found")
        raise err

if __name__ == "__main__":
    df = import_test()
    eda_test(df)
    df_encoded= encoder_helper_test(df)
    train_test_data = perform_feature_engineering_test(df_encoded)
    test_train_models(train_test_data)