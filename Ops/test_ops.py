import unittest
from unittest.mock import patch, MagicMock
import os
from Ops import Ops
import shutil
import json
import numpy as np
from PIL import Image

class TestOps(unittest.TestCase):

    @patch('Ops.Config')
    def setUp(self, MockConfig):
        # Mocking the config object
        self.mock_config = MockConfig.return_value
        self.mock_config.get.side_effect = lambda key: {
            "input_dir": "test_input",
            "MASK_EXTS": ["png"],
            "ORIGINAL_EXTS": ["jpg"],
            "classe": {"class1": 1, "class2": 2},
            "unif_ext": "png",
            "coco_output": "test_coco_output",
            "yolo_dset": "test_yolo_dset",
            "coco_format": "test_coco_format"
        }[key]
        
        self.ops = Ops("dummy_path")

    @patch('os.listdir')
    @patch('os.remove')
    @patch('PIL.Image.open')
    @patch('PIL.Image.Image.save')
    def test_convert_images(self, mock_save, mock_open, mock_remove, mock_listdir):
        # Mocking os.listdir
        mock_listdir.return_value = ['image1.jpg', 'image2.png']

        # Mocking Image.open to return a mock image
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image

        img_dir = "test_images"
        output_format = "png"
        self.ops.convert_images(img_dir, output_format)

        mock_open.assert_called()
        mock_save.assert_called()
        mock_remove.assert_called_once_with(os.path.join(img_dir, 'image1.jpg'))

    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch.object(Ops, 'convert_images')
    def test_get_img_mask_path(self, mock_convert_images, mock_isdir, mock_listdir):
        mock_listdir.side_effect = [
            ['image1.png', 'image2.png'],  # images directory
            ['class1', 'class2'],         # masks directory
            ['mask1.png', 'mask2.png'],   # class1 directory
            ['mask3.png', 'mask4.png']    # class2 directory
        ]
        mock_isdir.return_value = True
        
        imgs_paths, msks_paths = self.ops.get_img_mask_path()
        
        self.assertEqual(len(imgs_paths), 2)
        self.assertEqual(len(msks_paths), 4)

    @patch('cv2.findContours')
    def test_mask_to_polygons(self, mock_findContours):
        mock_findContours.return_value = ([
            np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[3, 3]]])
        ], None)

        mask = np.zeros((5, 5), dtype=np.uint8)
        polygons = self.ops.mask_to_polygons(mask)
        
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0]), 8)

    @patch('cv2.imread')
    @patch('shutil.copy')
    @patch('os.path.basename')
    @patch('os.path.dirname')
    @patch('os.makedirs')
    def test_process_masks(self, mock_makedirs, mock_dirname, mock_basename, mock_copy, mock_imread):
        mock_imread.side_effect = [np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10), dtype=np.uint8)]
        mock_dirname.return_value = 'class1'
        mock_basename.side_effect = lambda x: os.path.split(x)[-1]

        image_paths = ['image1.png']
        mask_paths = ['mask1.png']

        self.ops.process_masks(image_paths, mask_paths)

        mock_copy.assert_called()
        self.assertTrue(os.path.exists(os.path.join(self.ops.coco_output, 'coco_annotations.json')))
        with open(os.path.join(self.ops.coco_output, 'coco_annotations.json'), 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data['images']), 1)
            self.assertEqual(len(data['annotations']), 1)

    @patch('os.listdir')
    @patch('shutil.copy')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.load')
    def test_convert_to_yolo(self, mock_json_load, mock_open, mock_makedirs, mock_copy, mock_listdir):
        mock_json_load.return_value = {
            "images": [{"id": 1, "file_name": "image1.png", "width": 100, "height": 100}],
            "annotations": [{"image_id": 1, "category_id": 1, "segmentation": [[0, 0, 10, 10, 20, 20, 30, 30]]}]
        }
        mock_listdir.return_value = ['image1.png']

        self.ops.convert_to_yolo('dataset', 'annotations.json')

        mock_copy.assert_called()
        mock_open.assert_called()
        mock_makedirs.assert_called()
        self.assertTrue(os.path.exists(os.path.join(self.ops.yolo_dset, 'dataset', 'labels', 'image1.txt')))

if __name__ == '__main__':
    unittest.main()
