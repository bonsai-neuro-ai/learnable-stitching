set -o allexport
source .env
set +o allexport

mlflow server \
    --backend-store-uri mysql+pymysql://${MLFLOW_SQL_USERNAME}:${MLFLOW_SQL_PASSWORD}@localhost:3306/${MLFLOW_SQL_DATABASE} \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port 5001