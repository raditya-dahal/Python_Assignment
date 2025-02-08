import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def regression_analysis():
    file_path = "weight-height.csv"
    df = pd.read_csv(file_path)

    X = df[['Height']].values
    y = df['Weight'].values

    plt.scatter(X, y, alpha=0.5)
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs Weight')
    plt.show()

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.scatter(X, y, alpha=0.5, label='Actual Data')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Linear Regression on Height-Weight Data')
    plt.legend()
    plt.show()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f'RMSE: {rmse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

# Call the function
regression_analysis()
