import argparse
from pathlib import Path
import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

parser = argparse.ArgumentParser("projection_pr")
parser.add_argument("--data_pr_cc", type=str, help="Path data pr bajo escenario CC")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--projected_downscaling_pr", type=str, help="Path of projected pr output")

args = parser.parse_args()

lines = [
    f"Data pr cc: {args.data_pr_cc}",
    f"Model path: {args.model_input}",
    f"Downscaling pr ssp: {args.projected_downscaling_pr}"
]

for line in lines:
    print(line)

# Read test data
pr_cc_data = pd.read_csv(args.data_pr_cc)
sc = MaxAbsScaler()
pr_cc_sc = sc.fit_transform(pr_cc_data)
pr_cc_sc = pd.DataFrame(pr_cc_sc,columns=pr_cc_data.columns)


# Load model
folder = args.model_input
model  = pickle.load(open(f'{folder}/model.pkl', "rb"))
y_pred = model.predict(pr_cc_sc)

# Dataframe y_pred
projected_pr_data = pd.DataFrame(data=y_pred, columns=['pr_predict_cc'])

projected_pr_data.to_csv(Path(args.projected_downscaling_pr) / "downscaling_pr.csv")

