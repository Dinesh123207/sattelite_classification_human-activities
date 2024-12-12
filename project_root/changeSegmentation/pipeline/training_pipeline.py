# training_pipeline.py

from changeSegmentation.utils.main_utils import decodeImage
import os

class TrainPipeline:
    def __init__(self):
        self.model_path = "artifacts/model_trainer/best.pt"

    def run_pipeline(self):
        print("Training the model...")
        # Implement your training pipeline here
        # For example, you can use a training loop or call an external training script.
        # Placeholder for training logic
        os.system("yolo task=segment mode=train model=artifacts/model_trainer/best.pt")

        print(f"Model trained and saved to {self.model_path}")
