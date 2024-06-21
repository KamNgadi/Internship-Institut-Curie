# Librairies and frameworks
import os, sys
import cv2
import numpy as np
import json
import shutil
import yaml
from PIL import Image

# Add the absolute pathof Configs_file module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Configs_file.Configs import Config




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

    

    def __init__(self, config_path):
        # Get the configuration file from config object instantiation
        self.config = Config(config_path)
        self.input_dir = self.config.get("input_dir")
        self.mask_exts = self.config.get("MASK_EXTS")
        self.original_exts = self.config.get("ORIGINAL_EXTS")
        self.categories = self.config.get("classe") # Different object categories
        self.unf_ext = self.config.get("unif_ext") # Define the output extension for uniformity
        self.coco_output = self.config.get("coco_output")
        self.yolo_dset = self.config.get("yolo_dset")
        self.coco_format = self.config.get("coco_format")
    
    def convert_images(self, img_dir, output_format):
        """Convert all images in the given directory to the specified format, replacing existing files."""
        supported_exts = set(self.original_exts + self.mask_exts)
        
        for filename in os.listdir(img_dir):
            file_ext = filename.split('.')[-1].lower()
            if file_ext in supported_exts:
                img_path = os.path.join(img_dir, filename)
                img = Image.open(img_path)
                
                output_filename = f"{os.path.splitext(filename)[0]}.{output_format}"
                output_path = os.path.join(img_dir, output_filename)
                img.save(output_path)
                
                if file_ext != output_format:
                    os.remove(img_path)
                
                print(f"Converted {filename} to {output_filename}")
    
    def get_img_mask_path(self):
        """Retrieve image and mask paths from a given directory structure."""
        imgs_paths = []
        msks_paths = []

        imgs_dir = os.path.join(self.input_dir, 'images')
        msks_dir = os.path.join(self.input_dir, 'masks')

        self.convert_images(imgs_dir, self.unf_ext)
        for subdir in os.listdir(msks_dir):
            subdir_path = os.path.join(msks_dir, subdir)
            if os.path.isdir(subdir_path):
                self.convert_images(subdir_path, self.unf_ext)

        for file in os.listdir(imgs_dir):
            if file.split('.')[-1].lower() == self.unf_ext:
                imgs_paths.append(os.path.join(imgs_dir, file))

        for subdir in os.listdir(msks_dir):
            subdir_path = os.path.join(msks_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.split('.')[-1].lower() == self.unf_ext:
                        msks_paths.append(os.path.join(subdir_path, file))
                    
        return imgs_paths, msks_paths

    def mask_to_polygons(self, mask):
        """Draw contours around the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) > 2:
                poly = contour.reshape(-1).tolist()
                if len(poly) > 4:
                    polygons.append(poly)
        return polygons
    
    def process_masks(self, image_paths, mask_paths):
        """Use mask_to_polygons to get some properties about masks useful to convert them into COCO format."""
        annotations = []
        images = []
        image_id = 0
        ann_id = 0
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            image_id += 1
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
            if img is None:
                print(f"Could not read image file: {img_path}")
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            shutil.copy(img_path, os.path.join(self.coco_output, os.path.basename(img_path)))
            
            images.append({
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "height": img.shape[0],
                "width": img.shape[1]
            })
            
            unique_values = np.unique(mask)
            for value in unique_values:
                if value == 0:
                    continue
                
                object_mask = (mask == value).astype(np.uint8) * 255
                polygons = self.mask_to_polygons(object_mask)
                
                for poly in polygons:
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": self.categories.get(os.path.basename(os.path.dirname(mask_path)), 1),
                        "segmentation": [poly],
                        "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),
                        "bbox": list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),
                        "iscrowd": 0
                    })
        
        coco_output = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": cat_id, "name": cat_name} for cat_name, cat_id in self.categories.items()]
        }
        
        with open(os.path.join(self.coco_format, 'coco_annotations.json'), 'w') as f:
            json.dump(coco_output, f)

    def convert_to_yolo(self, dset, coco_annot_file):
        """Convert images to YOLO format."""
        input_images_path = os.path.join(self.coco_format, dset)
        input_json_path = os.path.join(self.coco_format, dset, coco_annot_file)
        output_images_path = os.path.join(self.yolo_dset, dset, "images")
        output_labels_path = os.path.join(self.yolo_dset, dset, "labels")
        
        with open(input_json_path) as f:
            data = json.load(f)

        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_labels_path, exist_ok=True)

        file_names = []
        for filename in os.listdir(input_images_path):
            if filename.split('.')[-1].lower() == self.unf_ext:
                source = os.path.join(input_images_path, filename)
                destination = os.path.join(output_images_path, filename)
                shutil.copy(source, destination)
                file_names.append(filename)

        def get_img_ann(image_id):
            return [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        def get_img(filename):
            return next((img for img in data['images'] if img['file_name'] == filename), None)

        for filename in file_names:
            img = get_img(filename)
            img_id = img['id']
            img_w = img['width']
            img_h = img['height']
            img_ann = get_img_ann(img_id)

            if img_ann:
                with open(os.path.join(output_labels_path, f"{os.path.splitext(filename)[0]}.txt"), "a") as file_object:
                    for ann in img_ann:
                        current_category = ann['category_id'] - 1
                        polygon = ann['segmentation'][0]
                        normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                        file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")
    

    
