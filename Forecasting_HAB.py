# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:33:19 2023

@author: Mostafa
"""
## 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


# Import statsmodels library for VIF calculation
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#resize matplot
plt.rcParams["figure.figsize"] = (7,7)

# Load the data from csv file
data = pd.read_csv('sorted_data.csv')

# Drop the non-numeric column
data = data.drop('Month', axis=1)

# Handle missing values with imputation using mean, median, mode, iterative imputation and KNN imputation
impute_methods = ['mean', 'median', 'most_frequent', 'iterative', 'knn']
imputed_datasets = []

for method in impute_methods:
    if method == 'iterative':
        imputer = IterativeImputer(random_state=0, max_iter=10)
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        imputer = SimpleImputer(missing_values=np.nan, strategy=method)
    imputed_data = imputer.fit_transform(data)
    imputed_datasets.append(pd.DataFrame(imputed_data, columns=data.columns))

# Plot the correlation matrix using heatmap
corr_matrix = data.corr()
N = 8
top_N_features = corr_matrix.nlargest(N+1, 'CYANOTOT')['CYANOTOT'].index
top_N_features = top_N_features.drop(['CYANOTOT', 'CYANOTOX'])
top_N_corr_matrix = data[top_N_features].corr()
sns.heatmap(top_N_corr_matrix, cmap='coolwarm', annot=True)
plt.title(f'Top {N} Most Correlated Features')
plt.savefig("corr_matrix.png", dpi=600)
plt.show()

# Train and test the regression models on each imputed dataset
for idx, imputed_data in enumerate(imputed_datasets):
    # Extract features and target variable
    X = imputed_data.drop(['CYANOTOT', 'CYANOTOX'], axis=1)
    y = imputed_data['CYANOTOT']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale the data using StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    

    # Use different regressions and find the suitable one
    models = [
        SVR(kernel='rbf'), 
        MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=95),
        DecisionTreeRegressor(random_state=0), 
        RandomForestRegressor(random_state=0)
    ]
    model_names = [
        'SVR', 
        'MLP',
        'DT',
        'RF'
    ]
    scores = []
    mse_scores = []
    mae_scores = []
    rmse_scores = []
    predictions = []

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        #Calculate the R^2
        score = r2_score(y_test, y_pred)
        scores.append(score)

        
        # Calculate and print mean squared error (MSE) for each model
        mse_score = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse_score)

        # Calculate and print MAE for each model
        mae_score = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae_score)

        # Calculate and print RMSE for each model
        rmse_score = np.sqrt(mse_score)
        rmse_scores.append(rmse_score)


        # Store the predictions for later use
        predictions.append(y_pred)

        # Print the coefficients and intercepts of the model
        print(f"Imputation Method: {impute_methods[idx]}")
        print("Model: ", model)
        if hasattr(model, 'coef_'):
            print("Coefficients: ", model.coef_)
        if hasattr(model, 'intercept_'):
            print("Intercept: ", model.intercept_)


        # Print the Nash-Sutcliffe Efficiency score of the model
        print("R^2: ", score)
        print("\n")
        
        print("MSE: ", mse_score)
        print("\n")
        
        print("MAE: ", mae_score)
        print("\n")
        
        print("RSME: ", rmse_score)
        print("\n")
        

    # Calculate and print the VIF for each predictor
    print(f"Imputation Method: {impute_methods[idx]}")
    print("VIF for each predictor:")
    vif = []
    for i in range(X.shape[1]):
        vif.append(variance_inflation_factor(X.values, i))
    for i, col in enumerate(X.columns):
        print(f"{col}: {vif[i]}")
    print("\n")

results = pd.DataFrame({'Model': model_names, 'R^2': scores, 'MSE': mse_scores, 'MAE': mae_scores, 'RMSE': rmse_scores})
print(results)

# Plot R^2 results in a bar chart
sns.barplot(x='Model', y='R^2', data=results)
plt.ylim(0, None)
plt.title(f"Imputation Method: {impute_methods[idx]} - Root Mean Squared Error")
plt.xticks(rotation=0)
plt.savefig(f"R2_{impute_methods[idx]}.png", dpi=600)
plt.show()

# Plot MSE  results in a bar chart
sns.barplot(x='Model', y='MSE', data=results)
plt.ylim(0, None)
plt.title(f"Imputation Method: {impute_methods[idx]} - Mean Squared Error")
plt.xticks(rotation=0)
plt.savefig(f"mse_{impute_methods[idx]}.png", dpi=600)
plt.show()

# Plot MAE results in a bar chart
sns.barplot(x='Model', y='MAE', data=results)
plt.ylim(0, None)
plt.title(f"Imputation Method: {impute_methods[idx]} -Mean Absolute Error")
plt.xticks(rotation=0)
plt.savefig(f"mae_{impute_methods[idx]}.png", dpi=600)
plt.show()

# Plot RMSE results in a bar chart
sns.barplot(x='Model', y='RMSE', data=results)
plt.ylim(0, None)
plt.title(f"Imputation Method: {impute_methods[idx]} - Root Mean Squared Error")
plt.xticks(rotation=0)
plt.savefig(f"rmse_{impute_methods[idx]}.png", dpi=600)
plt.show()

# Save the predictions for later use
results.to_csv(f"predictions_{impute_methods[idx]}.csv", index=False)
