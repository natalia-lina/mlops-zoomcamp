import mlflow
from os import listdir
from subprocess import run

# mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts
if __name__ == "__main__":
    print("MLFlow version: %s" % mlflow.__version__)

    run(["python", "preprocess_data.py", "--raw_data_path", ".", "--dest_path", "./output"])
    print("Number of output files: %d" % len(listdir("output")))