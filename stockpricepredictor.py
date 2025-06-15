import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download historical stock data
data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
data = data[['Close']]

# Step 2: Create feature and target
data['Prev_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

X = data[['Prev_Close']]
y = data['Close']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 6: Visualize predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, y_pred, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.title('AAPL Stock Price Prediction (Linear Regression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Print metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
