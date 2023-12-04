import argparse
from pathlib import Path
import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
import mlflow

# Optimizacion de parametros

parser = argparse.ArgumentParser("train_support_vector_machine")
parser.add_argument("--data_train", type=str, help="Path to training data")
parser.add_argument("--kernel", type=str, help="The function to measure the quality of a split.")
parser.add_argument("--model_output_svm_pickle", type=str, help="Path of output model pickle")

args = parser.parse_args()

lines = [
    f"data_train: {args.data_train}",
    f"kernel: {args.kernel}",
    f"model_output_svm_pickle: {args.model_output_svm_pickle}",
]

for line in lines:
    print(line)

# Read data
data_train = pd.read_csv(args.data_train)
X_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]

# Train model
model_SVM = SVR(kernel= args.kernel)

# Definir los hiperparámetros y sus posibles valores
param_grid = {
    'C': [100,  1000],
    'gamma' : [0.01 , 0.5, 1.],
}

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(estimator=model_SVM, param_grid=param_grid, 
                           cv=3)

# Ajustar el modelo con GridSearchCV
grid_search.fit(X_train, y_train)

# Obtener el modelo con el mejor rendimiento
best_model_SVM = grid_search.best_estimator_


# Save model.pkl
new_dir = Path('.', args.model_output_svm_pickle)
with open(f'{new_dir}/model.pkl', 'wb') as f:  # open a text file
    pickle.dump(best_model_SVM, f)
