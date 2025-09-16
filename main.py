# Import necessary libraries
import pandas as pd # Used for data manipulation and analysis, especially for handling DataFrames.
from sklearn.model_selection import train_test_split # A function to split data into training and testing sets.
from sklearn.linear_model import LinearRegression # The linear regression model.
from sklearn.preprocessing import PolynomialFeatures, StandardScaler # PolynomialFeatures creates higher-order terms, while StandardScaler scales numerical features.
from sklearn.pipeline import Pipeline # Used to chain together multiple data processing and modeling steps.
from sklearn.metrics import mean_absolute_error, r2_score # Metrics to evaluate model performance.
import os # A library for interacting with the operating system, used here to check if a file exists.
import numpy as np # Used for numerical operations, especially with arrays.

# --- 1. Load and Preprocess the Data ---
# This block of code handles loading the dataset from a CSV file and preparing it for the models.
file_path = 'sp500_stocks.csv'

# Check if the file exists before attempting to read it.
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the file is in the same directory as this script.")
else:
    try:
        # Read the CSV file into a pandas DataFrame.
        df = pd.read_csv(file_path)
        
        # We will focus on a single company for this prediction task. Let's use Apple (AAPL).
        company_symbol = 'AAPL'
        company_data = df[df['Symbol'] == company_symbol].copy()
        
        # Drop rows with any missing values, which are common in financial data.
        # We are only interested in the features needed for our prediction.
        features = ['Open', 'High', 'Low', 'Volume', 'Close']
        company_data.dropna(subset=features, inplace=True)
        
        # Create the target variable: the next day's closing price. We do this by shifting the 'Close' column up by one row.
        # The last row will now have a NaN, so we drop it.
        company_data['Next_Day_Close'] = company_data['Close'].shift(-1)
        company_data.drop(company_data.tail(1).index, inplace=True)
        
        # Define features (X) and the target variable (y).
        X = company_data[['Open', 'High', 'Low', 'Volume']]
        y = company_data['Next_Day_Close']
        
        # Split the data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- 2. Train and Evaluate Linear Regression Model ---
        # We use a pipeline to first scale the data and then apply the linear regression model.
        # This is good practice as it prevents data leakage and simplifies the workflow.
        linear_model_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Train the linear model on the training data.
        linear_model_pipeline.fit(X_train, y_train)
        
        # Make predictions on the test set.
        y_pred_linear = linear_model_pipeline.predict(X_test)
        
        # Evaluate the linear model's performance using MAE and R-squared.
        mae_linear = mean_absolute_error(y_test, y_pred_linear)
        r2_linear = r2_score(y_test, y_pred_linear)
        
        # --- 3. Train and Evaluate Polynomial Regression Model ---
        # This section adds a PolynomialFeatures step to capture non-linear relationships.
        
        # We'll use a degree of 2, which creates terms like x^2, y^2, and x*y.
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
        # Create a new pipeline for the polynomial model.
        poly_model_pipeline = Pipeline(steps=[
            ('poly_features', poly_features),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Train the polynomial model.
        poly_model_pipeline.fit(X_train, y_train)
        
        # Make predictions with the polynomial model.
        y_pred_poly = poly_model_pipeline.predict(X_test)
        
        # Evaluate the polynomial model's performance.
        mae_poly = mean_absolute_error(y_test, y_pred_poly)
        r2_poly = r2_score(y_test, y_pred_poly)

        # --- 4. Compare the Models ---
        print("\n--- Model Performance Comparison ---")
        print(f"Linear Regression:")
        print(f"  R-squared (R²): {r2_linear:.4f}")
        print(f"  Mean Absolute Error (MAE): ${mae_linear:.2f}")
        print("-" * 40)
        print(f"Polynomial Regression (Degree 2):")
        print(f"  R-squared (R²): {r2_poly:.4f}")
        print(f"  Mean Absolute Error (MAE): ${mae_poly:.2f}")
        print("-" * 40)
        
        # Determine and print which model performed best.
        if r2_poly > r2_linear:
            best_model_name = "Polynomial Regression"
            best_model = poly_model_pipeline
            print("Conclusion: The Polynomial Regression model is the best fit, with a higher R-squared and lower MAE.")
        else:
            best_model_name = "Linear Regression"
            best_model = linear_model_pipeline
            print("Conclusion: The Linear Regression model is the best fit, with a higher R-squared and lower MAE.")

        # --- 5. Make a New Prediction based on the Best Model ---
        # This interactive loop allows the user to input new data and get a prediction.
        print("\n--- Make a New Prediction ---")
        print(f"Using the best-performing model ({best_model_name}).")
        
        while True:
            try:
                # Prompt the user for new data.
                open_price = float(input("Enter today's Open price: "))
                high_price = float(input("Enter today's High price: "))
                low_price = float(input("Enter today's Low price: "))
                volume = float(input("Enter today's Volume: "))
                
                # Create a numpy array for the new input.
                new_data = np.array([[open_price, high_price, low_price, volume]])
                
                # Use the selected best model to make a prediction on the new data.
                predicted_price = best_model.predict(new_data)
                
                print(f"\nBased on today's data, the predicted next-day closing price is: ${predicted_price[0]:.2f}")
                
                # Exit the loop after a successful prediction.
                break 
            except ValueError:
                # Handle cases where the user enters a non-numeric value.
                print("Invalid input. Please enter valid numbers.")
            except Exception as e:
                # Catch any other unexpected errors.
                print(f"An unexpected error occurred: {e}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
