import pandas as pd
import numpy as np
import logging
import os
import warnings
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

print("Libraries imported successfully")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# to get arguments from command

parser=argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required=False,default=0.5)
parser.add_argument("--l1_ratio",type=float,required=False,default=0.5)
args=parser.parse_args()

print("Arg params are set")

# Evaluation Method

def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)


# Loading the data

data=pd.read_csv("D:\\Linear-Model-with-MLFlow\\red-wine-quality.csv")
data = data.drop('Unnamed: 0',axis=1)

print("Data ingestion completed")
# Separating independent and dependent variable : 

x = data.drop(['quality'],axis=1)
y = data[['quality']]

print("Data is divided into training and testing")

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

alpha = args.alpha
l1_ratio = args.l1_ratio
exp = mlflow.set_experiment(experiment_name="exp_1")

with mlflow.start_run(experiment_id=exp.experiment_id):
    print("Model training started")
    reg=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    reg.fit(x_train,y_train)

    print("Model training completed and prediction started")
    pred = reg.predict(x_test)

    print("Model is being evaluated")
    (rmse,mae,r2)=eval_metrics(y_test,pred)

    print(f"The Elasticnet model alpha: {alpha} and l1_ratio: {l1_ratio}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2_SCORE: {r2}")

    mlflow.log_param("alpha",alpha)
    mlflow.log_param("l1_ratio",l1_ratio)

    mlflow.log_metric("RMSE",rmse)
    mlflow.log_metric("MAE",mae)
    mlflow.log_metric("R2_SCORE",r2)

    mlflow.sklearn.log_model(reg,"mymodel")


