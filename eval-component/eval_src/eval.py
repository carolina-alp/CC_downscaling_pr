import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser("eval")
parser.add_argument("--score_result", type=str, help="Path of scoring result")
parser.add_argument("--eval_output", type=str, help="Path of output evaluation result")

args = parser.parse_args()

lines = [
    f"Eval result path: {args.score_result}",
    f"Evaluation output path: {args.eval_output}",
]

for line in lines:
    print(line)

# Read test and prediction data
data   = pd.read_csv(args.score_result)
y_test = np.array(data.iloc[:,0])
y_pred = np.array(data.iloc[:,1])

# Performance 
rep_r2_score = r2_score(y_test, y_pred)
rep_mean_absolute_error = mean_absolute_error(y_test, y_pred)
rep_mean_squared_error = mean_squared_error(y_test, y_pred)
values = [rep_r2_score,
          rep_mean_absolute_error,
          rep_mean_squared_error]
metrics = ['r2_score',
           'mean_absolute_error',
           'mean_squared_error'
            ]
df_rep = pd.DataFrame(values,index=metrics)

df_rep.to_csv(Path(args.eval_output) / "report.csv")

