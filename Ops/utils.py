import os
import re
import numpy as np
import pandas as pd
import json
import yaml
from sklearn.model_selection import train_test_split
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
from operator import is_not
from functools import partial
import PIL.Image
import ultralytics 
from ultralytics import YOLO
from IPython.display import display, Image
from IPython import display
import glob
import os
import random
import os
import glob
import cv2
import json
import shutil
# Label IDs of the dataset representing different categories


def convert_images(data_dir, input_format, unf_format):
        """Convert all images in the given directory to the specified format, replacing existing files.
        
        Args:
            img_dir: 
        """
        
        supported_exts = set(input_format)
        for filename in os.listdir(data_dir):
            file_ext = filename.split('.')[-1].lower()
            if file_ext in supported_exts:
                img_path = os.path.join(data_dir, filename)
                img = PIL.Image.open(img_path)
                
                output_filename = f"{os.path.splitext(filename)[0]}.{unf_format}"
                output_path = os.path.join(data_dir, output_filename)
                img.save(output_path)
                
                if file_ext != unf_format:
                    os.remove(img_path)
                
                #print(f"Converted {filename} to {output_filename}")

def get_img_mask_path(input_dir:str, img_format:str) -> list:
        
        """Retrieve image and mask paths from a given directory structure.
        
        Args:
            input_dir: repository's path containing images and masks
            img_format: images (masks and images format)

        Return: List for every single images and masks path
        """


        imgs_paths = []
        msks_paths = []

        imgs_dir = os.path.join(input_dir, 'images')
        msks_dir = os.path.join(input_dir, 'masks')

         # Check if the directories exist
        if not os.path.isdir(imgs_dir):
            raise Exception(f"The directory {imgs_dir} does not exist. Please ensure the directory is named 'images'.")
        if not os.path.isdir(msks_dir):
            raise Exception(f"The directory {msks_dir} does not exist. Please ensure the directory is named 'masks'.")

        #convert_images(imgs_dir, unf_ext)
        #for subdir in os.listdir(msks_dir):
        #    subdir_path = os.path.join(msks_dir, subdir)
        #    if os.path.isdir(subdir_path):
        #        convert_images(subdir_path, unf_ext)

        for file in os.listdir(imgs_dir):
            if file.split('.')[-1].lower() == img_format:
                imgs_paths.append(os.path.join(imgs_dir, file))

        for subdir in os.listdir(msks_dir):
            subdir_path = os.path.join(msks_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.split('.')[-1].lower() == img_format:
                        msks_paths.append(os.path.join(subdir_path, file))
                    
        return imgs_paths, msks_paths
        
def display_images_with_annotations(image_paths, annotation_paths):

    """
        Displays images with overlaid annotations.

        This function takes paths to images and their corresponding annotation files, and displays the images in a grid with the annotations drawn as polygons.

        Args:
            image_paths (list of str): A list of file paths to the images.
            annotation_paths (list of str): A list of file paths to the annotation files. Each file contains polygon coordinates and category IDs.

        The annotation file should have the following format:
            - Each line represents a single polygon.
            - The first value on each line is the category ID.
            - The remaining values are the normalized coordinates of the polygon vertices (x1, y1, x2, y2, ..., xn, yn).

        Example:
            annotation file line format:
            "0 0.1 0.2 0.3 0.4 0.5 0.6"

        Returns:
            None
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, img_path, ann_path in zip(axs.ravel(), image_paths, annotation_paths):
        # Load image using OpenCV and convert it from BGR to RGB color space
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
        
        ax.imshow(image)
        ax.axis('off')  # Turn off the axes

        # Open the annotation file and process each line
        with open(ann_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                category_id = int(parts[0])
                color = tuple(np.random.rand(3))  # Generate a random RGB color
                polygon = [float(coord) for coord in parts[1:]]
                polygon = [coord * img_w if i % 2 == 0 else coord * img_h for i, coord in enumerate(polygon)]
                polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                patch = patches.Polygon(polygon, closed=True, edgecolor=color, fill=False)
                ax.add_patch(patch)

    plt.tight_layout()
    plt.show()

# Get all image files
image_dir = "yolo_dataset/train/images/"
annotation_dir = "yolo_dataset/train/labels/"
all_image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png'))]
random_image_files = random.sample(all_image_files, 4)

# Get corresponding annotation files
image_paths = [os.path.join(image_dir, f) for f in random_image_files]
annotation_paths = [os.path.join(annotation_dir, f.replace(".tif", ".txt")) for f in random_image_files]

def convert_to_yolo(input_images_path, input_json_path, output_images_path, output_labels_path):

    """
        Converts image annotations from COCO format to YOLO format and copies images to a new directory.

        Args:
            input_images_path (str): Path to the directory containing input images.
            input_json_path (str): Path to the JSON file containing COCO annotations.
            output_images_path (str): Path to the directory where output images will be saved.
            output_labels_path (str): Path to the directory where YOLO format label files will be saved.

        Returns:
            None

        This function performs the following steps:
            1. Opens and reads the JSON file containing image annotations.
            2. Creates directories for output images and labels if they do not exist.
            3. Copies images from the input directory to the output directory.
            4. Extracts image annotations and converts them to YOLO format.
            5. Writes the normalized polygon data to text files in the output labels directory.

        Note:
            - The function expects the input JSON file to be in COCO format.
            - The function normalizes the polygon coordinates based on the image dimensions.
            - The output label files are named according to the corresponding image files with a .txt extension.

        Example:

                base_input_path = "input/"
                base_output_path = "yolo_dataset/"

                # Processing training dataset 
                convert_to_yolo(
                    input_images_path=os.path.join(base_input_path, "train_images"),
                    input_json_path=os.path.join(base_input_path, "train_images/coco_annotations.json"),
                    output_images_path=os.path.join(base_output_path, "train/images"),
                    output_labels_path=os.path.join(base_output_path, "train/labels")
                )
"""
    # Open JSON file containing image annotations
    f = open(input_json_path)
    data = json.load(f)
    f.close()

    # Create directories for output images and labels
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # List to store filenames
    file_names = []
    for filename in os.listdir(input_images_path):
        if filename.endswith((".tif", ".png")):
            source = os.path.join(input_images_path, filename)
            destination = os.path.join(output_images_path, filename)
            shutil.copy(source, destination)
            file_names.append(filename)

    # Function to get image annotations
    def get_img_ann(image_id):
        return [ann for ann in data['annotations'] if ann['image_id'] == image_id]

    # Function to get image data
    def get_img(filename):
        return next((img for img in data['images'] if img['file_name'] == filename), None)

    # Iterate through filenames and process each image
    for filename in file_names:
        img = get_img(filename)
        if img is None:
            print(f"Warning: No annotation found for image {filename}")
            continue  # Skip this image if no annotation is found
        
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        img_ann = get_img_ann(img_id)

        # Write normalized polygon data to a text file
        if img_ann:
            with open(os.path.join(output_labels_path, f"{os.path.splitext(filename)[0]}.txt"), "a") as file_object:
                for ann in img_ann:
                    current_category = ann['category_id'] - 1
                    polygon = ann['segmentation'][0]
                    normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                    file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")


def create_yaml(input_json_path, output_yaml_path, train_path, val_path, test_path=None):

    """
        Creates a YAML file for the dataset based on the input JSON file.

        Args:
            input_json_path (str): Path to the input JSON file containing dataset annotations.
            output_yaml_path (str): Path where the output YAML file will be saved.
            train_path (str): Path to the training dataset.
            val_path (str): Path to the validation dataset.
            test_path (str, optional): Path to the test dataset. Defaults to None.

        Returns:
            None

        Example:
                create_yaml(
                            input_json_path=os.path.join(base_input_path, "train_images/coco_annotations.json"),
                            output_yaml_path=os.path.join(base_output_path, "data.yaml"),
                            train_path="train/images",
                            val_path="valid/images",
                            test_path='../test/images'  # or None if not applicable
                            )
    """
    with open(input_json_path) as f:
        data = json.load(f)
    
    # Extract the category names
    names = [category['name'] for category in data['categories']]
    
    # Number of classes
    nc = len(names)

    # Create a dictionary with the required content
    yaml_data = {
        'names': names,
        'nc': nc,
        'test': test_path if test_path else '',
        'train': train_path,
        'val': val_path
    }

    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)





def process_masks(mask_paths: str, output_dir: str, category_ids:dict, MASKS_EXT, ORIGINAL_EXT) -> None:
  
    """
        Converts mask images to polygons and prepares them for COCO format conversion given the following project structure.
        The image and mask filenames should match.


            Project folder/   #Primary data folder for the project
        ├── input/           #All input data is stored here. 
        │   ├── train_images/
        │   │   ├── image01.png
        │   │   ├── image02.png
        │   │   └── ...
        │   ├── train_masks/        #All binary masks organized in respective sub-directories.
        │   │   ├── class1/
        │   │   │   ├── image01.png
        │   │   │   ├── image02.png
        │   │   │   └── ...
        │   │   ├── class2/
        │   │   │   ├── image01.png
        │   │   │   ├── image02.png
        │   │   │   └── ...
        │   │   ├── class3/
        │   │       ├── image01.png
        │   │       ├── image02.png
        │   │        ...           
        │   │       
        │   ├── val_images/
        │   │   ├── image05.png
        │   │   ├── image06.png
        │   │   └── ...
        │   └── val_masks/
        │       ├── class1/
        │       │   ├── image05.png
        │       │   ├── image06.png
        │       │   └── ...
        │       ├── class2/
        │       │   ├── image05.png
        │       │   ├── image06.png
        │       │   └── ...
        │       ├── class3/
        │           ├── image05.png
        │           ├── image06.png
        │           └── ...
        │       
        └── ...

        Args:
            mask_paths (str): Path to the directory containing mask images.
            output_dir (str): Path to the directory where the processed images and coco.json file will be saved.
            category_ids (dict): A dictionary mapping class names to their corresponding category IDs for segmentation.
            MASK_EXT (str): The file extension of the mask images.
            ORIGINAL_EXT (str): The file extension of the original images.

        Returns:
            None

        Example:
                HOME # Working directory
                train_masks_path= HOME +  "/input/train_masks"
                train_output_dir =HOME + "/input/train_images"
                process_masks(mask_paths=train_masks_path,
                            output_dir=train_output_dir,
                            category_ids={'cell':1, 'nuclei':2}, 
                            MASKS_EXT='png',
                            ORIGINAL_EXT='png')


    """
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
    #MASKS_EXT = 'tif'
    #ORIGINAL_EXT = 'tif'

    for category in category_ids.keys():
        for mask_image in glob.glob(os.path.join(mask_paths, category, f'*.{MASKS_EXT}')):
            img_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
            mask = cv2.imread(mask_image, cv2.IMREAD_UNCHANGED)
            
            if mask is None:
                print(f"Warning: Mask image {mask_image} could not be read.")
                continue
            
            # Get image dimensions
            height, width = mask.shape[:2]

            # Create or find existing image annotation
            if img_file_name not in map(lambda img: img['file_name'], images):
                image_id += 1
                images.append({
                    "id": image_id,
                    "file_name": img_file_name,
                    "height": height,
                    "width": width
                })
                image = images[-1]  # The newly added image
            else:
                image = [element for element in images if element['file_name'] == img_file_name][0]

            unique_values = np.unique(mask)
            for value in unique_values:
                if value == 0:
                    continue

                object_mask = (mask == value).astype(np.uint8) * 255
                polygons = mask_to_polygons(object_mask)

                for poly in polygons:
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": image['id'],
                        "category_id": category_ids[category],
                        "segmentation": [poly],
                        "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),
                        "bbox": list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),
                        "iscrowd": 0
                    })

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()]
    }

    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(coco_output, f)

    print("Created %d annotations for images in folder: %s" % (len(annotations), mask_paths))


def create_dir_structure(base_dir:str) ->None:

    """Create the directory structure for train and validation sets.
    Args:
        base_dir: Directory we want to host the train and validation sets.

    Return: None, but create folders to build desired structure


    """

    dirs = [
        os.path.join(base_dir, 'train_images'),
        os.path.join(base_dir, 'train_masks', 'cell'),
        os.path.join(base_dir, 'train_masks', 'nuclei'),
        os.path.join(base_dir, 'val_images'),
        os.path.join(base_dir, 'val_masks', 'cell'),
        os.path.join(base_dir, 'val_masks', 'nuclei')
    ]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def split_data(input_dir:str, output_dir:str, categories:list, val_ratio=0.2) -> None:

    """
        Splits the dataset into training and validation sets, creating the necessary directory structure and copying the images and masks accordingly.

        Args:
            input_dir (str): Path to the input directory containing 'images' and 'masks' subdirectories.
            output_dir (str): Path to the output directory where the split data will be saved.
            categories (list): List of mask categories (e.g., ['cell', 'nuclei']).
            val_ratio (float, optional): The ratio of validation data to total data. Default is 0.2.

        Returns:
            None

        Example:
                input_directory = "train"
                output_directory = HOME + "/input"
                split_data(input_directory, output_directory, categories=['cell', 'nuclei'])
    """
    create_dir_structure(output_dir)
    
    train_images_dir = os.path.join(input_dir, 'images')
    print(f"train_images_dir: {train_images_dir}")
    train_masks_dir = os.path.join(input_dir, 'masks')
    print(f"train_masks_dir: {train_masks_dir}")
    
    for category in categories:
        mask_category_dir = os.path.join(train_masks_dir, category)
        print(f"Processing category: {category}")
        
        if not os.path.exists(mask_category_dir):
            print(f"Mask directory for category '{category}' does not exist: {mask_category_dir}")
            continue
        
        image_files = sorted(os.listdir(train_images_dir))
        image_files = [f for f in image_files if f.lower().endswith(('tif', 'png'))]  # Make the extension check case-insensitive
        
        num_images = len(image_files)
        num_val = int(num_images * val_ratio)
        
        val_images = random.sample(image_files, num_val)
        train_images = [img for img in image_files if img not in val_images]
        
        for img in train_images:
            img_name = os.path.basename(img)
            src_mask_path = os.path.join(mask_category_dir, img)
            dst_mask_path = os.path.join(output_dir, 'train_masks', category, img)
            
            if not os.path.exists(src_mask_path):
                print(f"Mask file does not exist: {src_mask_path}")
                continue
            
            shutil.copy(src_mask_path, dst_mask_path)
        
        for img in val_images:
            img_name = os.path.basename(img)
            src_mask_path = os.path.join(mask_category_dir, img)
            dst_mask_path = os.path.join(output_dir, 'val_masks', category, img)

            if not os.path.exists(src_mask_path):
                print(f"Mask file does not exist: {src_mask_path}")
                continue

            shutil.copy(src_mask_path, dst_mask_path)

        print(f"Category {category} - Train: {len(train_images)}, Val: {len(val_images)}")

    for img in train_images:
        img_name = os.path.basename(img)
        src_img_path = os.path.join(train_images_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'train_images', img_name)
        shutil.copy(src_img_path, dst_img_path)

    for img in val_images:
        img_name = os.path.basename(img)
        src_img_path = os.path.join(train_images_dir, img_name)   
        dst_img_path = os.path.join(output_dir, 'val_images', img_name)
        shutil.copy(src_img_path, dst_img_path)


def seg_masks(data_dir: str, image_dir: str, model: str) -> None:

    """
Generates segmented masks for images using the specified model and saves them to a designated directory.

This function processes images from a specified directory using a given model to produce segmented masks. The segmented masks are saved to a subdirectory named 'seg_masks' within the `data_dir`.

Args:
    data_dir (str): Absolute path to the directory containing the images and where the segmented masks will be saved.
    image_dir (str): Relative path to the directory containing the images to be segmented.
    model (str): Path to the model used for inference.

Returns:
    None

Raises:
    OSError: If there is an issue creating the 'seg_masks' directory.
    cv2.error: If there is an issue reading an image.
    Exception: If the model fails to load or predict.

Processes:
1. Sets up the 'seg_masks' directory within `data_dir` to save the results.
2. Loads each image from `image_dir`, verifies it is a valid image file.
3. Loads the specified model and performs segmentation on each image.
4. Generates masks for each image, resizes them to match the original image dimensions, and overlays them.
5. Saves the resulting segmented masks to the 'seg_masks' directory with a naming convention that reflects the original images.
6. Handles errors gracefully, providing informative messages if something goes wrong.

Example:
    seg_masks('/path/to/data', 'images', '/path/to/model.pt')
"""

    # create segmented masks directory
    seg_masks_dir = os.path.join(data_dir, 'seg_masks')
    if not os.path.exists(seg_masks_dir):
        try:
            os.makedirs(seg_masks_dir, exist_ok=True)
        except os.error as e:
            print(f'Same directory exist:{e}')
            exit()
    # set the path to the image directory
    img_dir = os.path.join(data_dir, image_dir)

    # iterate over all files in the test image directory
    for file in os.listdir(img_dir):
        # contruct full file path
        file_path = os.path.join(img_dir, file)

        # check if the file is image
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.tif', '.tiff')):
            # read the image with OpenCV
            try:
                img = cv2.imread(file_path)
                h, w, _ = img.shape
            except cv2.error as e:
                print(f"Failed to load image: {e}")
                exit()

        # create the YOLOV8 
        try:
            mdel = YOLO(model)
        except Exception as e:
            print(f"Failed to load model: {e}")
            exit()
        
        # Perform the prediction
        img2 = cv2.resize(img, (640,640))
        #img2 = cv2.resize(img, (img.shape[1], img.shape[0]))
        #imgsz = (img2.shape[0], img2.shape[0])
        results = mdel.predict(img2, iou=.5)

        if(results[0].masks is not None):
            # Get the size of the original image (height, width, channels)
            h2, w2, c2 = results[0].orig_img.shape

            # Create a black image with the same size as the original image
            black_img = np.zeros_like(results[0].orig_img)
            black_img = black_img[:,:,0]
            
            # Create a copy of the original image to layer the masks on
            #layered_img = results[0].orig_img.copy()

            # Loop over all masks in the results
            for i, mask_raw in enumerate(results[0].masks):
                # Convert mask to single channel image
                mask_raw = mask_raw.cpu().data.numpy().transpose(1, 2, 0)

                # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
                mask = cv2.resize(mask_raw, (w2, h2))

                # Convert the mask to the correct data type
                mask = mask.astype(np.uint16)

                #multiply by i+1
                mask = mask*(i+1)

                #add mask to black_img
                black_img = np.maximum(black_img, mask)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
        
        name, ext = os.path.splitext(file)

        # set the path to the segmented mask with the same naming system like images
        seg_mask = os.path.join(seg_masks_dir, seg_masks_dir + '/' + name.replace("img", "masks") + "_seg." + ext)
        
        # Close all windows
        black_img = cv2.resize(black_img, (w,h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(seg_mask, black_img)
        cv2.destroyAllWindows()      


        







