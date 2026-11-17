import os
import mlflow
from dotenv import load_dotenv    
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
load_dotenv() 

user = os.getenv("MLFLOW_SQL_USERNAME")
password = os.getenv("MLFLOW_SQL_PASSWORD")
db_name = os.getenv("MLFLOW_SQL_DATABASE")

mlflow.set_tracking_uri(f"mysql+pymysql://{user}:{password}@localhost:3306/{db_name}")

for e in mlflow.search_experiments(view_type=ViewType.DELETED_ONLY):
    print(e.experiment_id, e.name)

#mlflow.restore_experiment("2")