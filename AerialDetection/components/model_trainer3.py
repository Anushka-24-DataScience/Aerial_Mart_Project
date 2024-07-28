import os
import sys
import yaml
import subprocess
from AerialDetection.utils.main_utils import read_yaml_file
from AerialDetection.logger import logging
from AerialDetection.exception import AppException
from AerialDetection.entity.config_entity import ModelTrainerConfig
from AerialDetection.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def execute_command(self, command: str):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            returncode, stdout, stderr = self.execute_command("unzip data.zip")
            if returncode != 0:
                raise AppException(f"Failed to unzip data.zip. Error: {stderr}", sys)
            os.remove("data.zip")

            # Rename data.yaml.yaml to data.yaml if it exists
            data_yaml_path = "data.yaml"
            if os.path.exists(f"{data_yaml_path}.yaml"):
                os.rename(f"{data_yaml_path}.yaml", data_yaml_path)

            # Ensure the data.yaml file is present
            if not os.path.exists(data_yaml_path):
                raise AppException("data.yaml file not found after unzipping", sys)

            with open(data_yaml_path, 'r') as stream:
                data_config = yaml.safe_load(stream)
                num_classes = str(data_config['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            logging.info(f"Model config file name: {model_config_file_name}")

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            config['nc'] = int(num_classes)

            custom_model_config_path = f'yolov5/models/custom_{model_config_file_name}.yaml'
            with open(custom_model_config_path, 'w') as f:
                yaml.dump(config, f)

            # Execute the training command
            train_command = (
                f"cd yolov5/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} "
                f"--epochs {self.model_trainer_config.no_epochs} --data ../{data_yaml_path} "
                f"--cfg ./models/custom_{model_config_file_name}.yaml --weights {self.model_trainer_config.weight_name} "
                f"--name yolov5s_results --cache"
            )
            returncode, stdout, stderr = self.execute_command(train_command)
            if returncode != 0:
                raise AppException(f"Training command failed. Error: {stderr}", sys)

            # Check if the training output file exists
            best_model_path = "yolov5/runs/train/yolov5s_results/weights/best.pt"
            if not os.path.exists(best_model_path):
                raise AppException(f"Training failed, best model file not found at {best_model_path}", sys)

            # Copy the trained model to the desired location
            os.system(f"cp {best_model_path} yolov5/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp {best_model_path} {self.model_trainer_config.model_trainer_dir}/")

            # Cleanup
            os.system("rm -rf yolov5/runs")
            os.system("rm -rf train valid")
            os.remove(data_yaml_path)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
