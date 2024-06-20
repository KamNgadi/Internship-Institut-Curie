# Librairies and frameworks
import os
from 

class Ops():

    """class to  organize input data, make transformations and any operations required 
      to make data suitable for a given model
        Suit for input directory with the folowing structure:

     ├── input/ #All input data is stored here.
        │ ├── images/
        │ │ ├── image01.png
        │ │ ├── image02.png
        │ │ └── ...
        │ ├── masks/ #All binary  or labeled masks organized in respective sub-directories.
        │ │ ├── classe1/
        │ │ │ ├── masks01.png
        │ │ │ ├── masks02.png
        │ │ │ └── ...
        │ │ ├── classe2/
        │ │ │ ├── masks01.png
        │ │ │ ├── masks02.png
        │ │ │ └── ...
        │ │ ├── classe3/
        │ │ │ ├── maskse01.png
        │ │ │ ├── masks02.png
        │ │ │ └── ...
        │ │ └── class3/
        │ │ ├── imasks01.png
        │ │ ├── masks02.png
        │ │ └── ...

        optional part
        │ ├── val_images/
        │ │ ├── masks05.png
        │ │ ├── masks06.png
        │ │ └── ...
        │ └── masks/
        │ ├── classe1/
        │ │ ├── masks05.png
        │ │ ├── imasks06.png
        │ │ └── ...
        │ ├── classe2/
        │ │ ├── masks05.png
        │ │ ├── masks06.png
        │ │ └── ...
        │ ├── classe3/
        │ │ ├── masks05.png
        │ │ ├── masks06.png
        │ │ └── ...
        │ └── classe34/
        │ ├── masks05.png
        │ ├── masks06.png
        │ └── ...
        └── ...
    """

    def __int__(self, config_path):
        self.input = 

    
    def get_img_mask_path(self, con):

        """Retrieve image and mask paths from a given directory structure.
        
        Args:
            data_dir (str): The root directory containing 'images' and 'masks' directories.
        
        Returns:
            list: List of image paths.
            list: List of corresponding mask paths."""
        
        self.data_dir = input_dir
        
        imgs_paths =[]
        msks_paths = []

        # Define the paths for images and masks directories
        imgs_dir = os.path.join(input_dir, 'images')
        msks_dir = os.path.join(input_dir, 'masks')

        # Check if the images directory exists
        if not os.path.exists(imgs_dir):
            print(f"Images directory does not exist: {imgs_dir}")
            return imgs_paths, msks_paths

        # Collect all image files in the images directory
        for file in os.listdir(imgs_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                imgs_paths.append(os.path.join(imgs_dir, file))

        # Collect all mask files from the subdirectories in the masks directory
        for subdir in os.listdir(msks_dir):
                subdir_path = os.path.join(msks_dir, subdir)
                if os.path.isdir(subdir_path):  # Ensure it's a directory
                    for file in os.listdir(subdir_path):
                        if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                            msks_paths.append(os.path.join(subdir_path, file))
                    
        return imgs_paths, msks_paths

