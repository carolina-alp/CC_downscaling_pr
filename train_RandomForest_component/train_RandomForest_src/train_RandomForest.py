import argparse
from pathlib import Path
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import mlflow

# Optimizacion de parametros

parser = argparse.ArgumentParser("train_random_forest")
parser.add_argument("--data_train", type=str, help="Path to training data")
parser.add_argument("--criterion", type=str, help="The function to measure the quality of a split.")
parser.add_argument("--model_output_rf_pickle", type=str, help="Path of output model pickle")

args = parser.parse_args()

lines = [
    f"data_train: {args.data_train}",
    f"criterion: {args.criterion}",
    f"model_output_rf_pickle: {args.model_output_rf_pickle}",
]

for line in lines:
    print(line)

# Read data
data_train = pd.read_csv(args.data_train)
X_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]

# Train model
model_RF = RandomForestRegressor(criterion= args.criterion)

# Definir los hiperparámetros y sus posibles valores
param_grid = {
    'n_estimators': [5,10,20],
    'max_depth' : [10,20,25],
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(estimator=model_RF, param_grid=param_grid, 
                           cv=3)

# Ajustar el modelo con GridSearchCV
grid_search.fit(X_train, y_train)

# Obtener el modelo con el mejor rendimiento
best_model_RF = grid_search.best_estimator_


# Save model.pkl
new_dir = Path('.', args.model_output_rf_pickle)
with open(f'{new_dir}/model.pkl', 'wb') as f:  # open a text file
    pickle.dump(best_model_RF, f)
