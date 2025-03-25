FROM ghcr.io/mlflow/mlflow:v2.21.0

WORKDIR /app

COPY start_mlflow.sh /app/start_mlflow.sh

ENTRYPOINT ["/usr/bin/env", "bash", "/app/start_mlflow.sh"]

EXPOSE 8088
