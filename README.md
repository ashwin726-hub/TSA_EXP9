## Ex.No: 6 HOLT WINTERS METHOD
## Date:23-04-2025

## AIM:

To implement Holt-Winters model on National stocks exchange Data Set and make future predictions

## ALGORITHM:

- You import the necessary libraries
- You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as - - - datetime, and perform some initial data exploration
- You group the data by date and resample it to a monthly frequency (beginning of the month
- You plot the time series data
- You import the necessary 'statsmodels' libraries for time series analysis
- You decompose the time series data into its additive components and plot them:
- You calculate the root mean squared error (RMSE) to evaluate the model's performance
- You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt- 
- Winters model to the entire dataset and make future predictions
- You plot the original sales data and the predictions

## PROGRAM :


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
data = pd.read_csv('/content/AirPassengers (1).csv', parse_dates=['date'], index_col='date')
print(data.head())

# Step 2: Resample to monthly data (if needed)
data_monthly = data.resample('MS').sum()  # MS = Month Start
print(data_monthly.head())

# Step 3: Plot the monthly data
data_monthly.plot()
plt.title('Monthly Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

# Step 4: Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

# Plot scaled data
scaled_data.plot()
plt.title('Scaled Monthly Data')
plt.show()

# Step 5: Seasonal Decomposition
decomposition = seasonal_decompose(data_monthly, model="additive", period=12)
decomposition.plot()
plt.show()

# Step 6: Prepare data for modeling
# Shift scaled data up by 1 (multiplicative needs all positive)
scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Step 7: Build the Exponential Smoothing model
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Step 8: Forecast on the test set
test_predictions_add = model_add.forecast(steps=len(test_data))

# Step 9: Visual evaluation
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["Train Data", "Predictions", "Test Data"])
ax.set_title('Visual Evaluation')
plt.show()

# Step 10: Model evaluation (RMSE)
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"Test RMSE: {rmse}")

# Print variance and mean
print("Variance of scaled data:", scaled_data.var())
print("Mean of scaled data:", scaled_data.mean())

# Step 11: Final Model Training and Future Forecast
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4))  # Forecasting next 1 year (quarter of dataset)

# Plot the final predictions
ax = data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["Actual Data", "Future Predictions"])
ax.set_xlabel('Date')
ax.set_ylabel('Number of Passengers')
ax.set_title('Final Forecast for Future')
plt.show()



## OUTPUT:

Scaled_data plot:
![Screenshot 2025-04-26 160214](https://github.com/user-attachments/assets/206c1f8e-f49a-442d-a8ac-90df87b539e8)


DECOMPOSED PLOT:
![Screenshot 2025-04-26 160230](https://github.com/user-attachments/assets/d8f4f3ab-24fa-4f47-a6f6-95f34e57c734)

Test Prediction:
![Screenshot 2025-04-26 160249](https://github.com/user-attachments/assets/2ae48f9d-a629-4c21-a0ba-503ad51f8bf7)

Model Performance metrices:
![Screenshot 2025-04-26 160308](https://github.com/user-attachments/assets/12f3c176-bc21-40e2-8d4d-4dae9f82a562)

Final prediciton:
![Screenshot 2025-04-26 160348](https://github.com/user-attachments/assets/bd846a9c-07d7-4f86-8b57-e6a2f659b123)


## RESULT:

Thus the program run successfully based on the Holt Winters Method model.
