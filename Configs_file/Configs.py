#Library
import yaml
from argparse import ArgumentParser

#  

class Config:

    """
    Manage configuration file with flexibility and modularity.
    
    """

    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config= yaml.safe_load(file)
        self.validate_config()

    def validate_config(self):

        """
        Check the required sections availability and her type
        """
        required_keys = {
            "project_dir": str,
            "datasets": str,
            "coco_style_dir": str,
            "test_size": float,
            "random_state": int,
            "yolo_style_dir": str

        }
        for key, expected_type in required_keys.items():
            if key not in self.config:
                raise ValueError(f'Missing configuration section: "{key}"')
            if not isinstance(self.config[key], expected_type):
                raise TypeError(f"Expected type for '{key}' is {expected_type.__name__}, but got {type(self.config[key]).__name__}")
            

    def get(self, key):

        """Get neeeded section"""

        if key not in self.config:
            raise ValueError(f"Missing required configuration section: '{key}'")
        return self.config[key]
    
    def parse_args():

        """Argument parser"""

        parser = ArgumentParser(description="YOLOv8 Evaluation")
        parser.add_argument('--config', type=str, required=True, help="Path to confuration file")
        return parser.parse_args()
        