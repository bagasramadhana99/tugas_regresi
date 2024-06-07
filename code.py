import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

def calculate_rmse(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse

def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, calculate_rmse(model, X_test, y_test)

def exponential_regression(X_train, X_test, y_train, y_test):
    X_train_exp = np.log(X_train)
    y_train_exp = np.log(y_train)
    model = LinearRegression()
    model.fit(X_train_exp, y_train_exp)
    return model, calculate_rmse(model, np.log(X_test), y_test)

def load_data():
    try:
        data = pd.read_csv("Student_Performance.csv")
        X = data[['Hours Studied']]
        y = data['Performance Index']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("File not found.")

def main():
    X_train, X_test, y_train, y_test = load_data()

    root = tk.Tk()
    root.title("Regression Analysis")

    label = tk.Label(root, text="Choose Regression Method:")
    label.pack()

    linear_button = tk.Button(root, text="Linear Regression", command=lambda: show_result(linear_regression(X_train, X_test, y_train, y_test)))
    linear_button.pack()

    exponential_button = tk.Button(root, text="Exponential Regression", command=lambda: show_result(exponential_regression(X_train, X_test, y_train, y_test)))
    exponential_button.pack()

    def show_result(result):
        model, rmse = result
        predictions = model.predict(X_test)
        
        plt.scatter(X_test, y_test, color='black', label='Actual')
        plt.plot(X_test, predictions, color='blue', linewidth=3, label='Predicted')
        plt.xlabel('Hours Studied')
        plt.ylabel('Performance Index')
        plt.title('Regression Result')
        plt.legend()
        plt.show()

        result_window = tk.Toplevel(root)
        result_window.title("Result")
        label_model = tk.Label(result_window, text=f"Model Coefficients: {model.coef_}")
        label_model.pack()
        label_rmse = tk.Label(result_window, text=f"Root Mean Squared Error: {rmse}")
        label_rmse.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
