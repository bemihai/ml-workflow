#!/usr/bin/env bash

set -e

echo "Starting MLFlow Tracking Server"

mlflow server \
  --host 0.0.0.0 \
  --port 8088

echo "MLFlow Tracking Server started at http://localhost:8080/"