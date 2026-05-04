
'''
In this file we are going to load the data and develop MLR code in oops concept
'''
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import sys
import warnings

warnings.filterwarnings("ignore")
import pickle


class MLR:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            self.df['city'] = self.df['city'].map({city: i for i, city in enumerate(self.df['city'].unique())})
            self.X = self.df.drop('price', axis=1)
            self.y = self.df['price']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)
            print(f"Training dataset size : {len(self.X_train)} : {len(self.y_train)}")
            print(f"Testing dataset size : {len(self.X_test)} : {len(self.y_test)}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")

    def training(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train, self.y_train)
            self.y_train_predictions = self.reg.predict(self.X_train)
            print(f"Train Accuracy : {r2_score(self.y_train, self.y_train_predictions)}")
            print(f"Train Loss : {root_mean_squared_error(self.y_train, self.y_train_predictions)}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")

    def testing(self):
        try:
            self.y_test_predictions = self.reg.predict(self.X_test)
            print(f"Test Accuracy : {r2_score(self.y_test, self.y_test_predictions)}")
            print(f"Test Loss : {root_mean_squared_error(self.y_test, self.y_test_predictions)}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")

    def check_own_data(self):
        try:

            print("Columns:", list(self.X.columns))

            test_data = [1200, 1800, 1900] + [0] * (len(self.X.columns) - 3)

            prediction = self.reg.predict([test_data])
            print(f"Prediction: {prediction[0]}")

        except Exception as e:
            er_type, er_obj, er_tb = sys.exc_info()
            print(f"Error at line {er_tb.tb_lineno}: {e}")




    def saving_model(self):
        try:
            with open("Model.pkl", "wb") as f:
                pickle.dump(self.reg, f)

            with open("columns.pkl", "wb") as f:
                pickle.dump(self.X.columns, f)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")


if __name__ == "__main__":
    try:
        path = "house prediction.csv"
        obj = MLR(path)
        obj.training()
        obj.testing()
        obj.check_own_data()
        obj.saving_model()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f"Error in Line No : {er_line.tb_lineno} : due to : {er_type} and reason was : {er_msg}")












