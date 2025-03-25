# mlflow-server
MLFlow tracking server deployed locally with Docker.  

## Usage

### Build the Docker image
```bash     
docker build -t mlflow-server .
``` 

### Run the Docker container
```bash 
docker run -d -p 8088:8088 mlflow-server
```

