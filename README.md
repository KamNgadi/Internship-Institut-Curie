## Abstract

Segmentation in imaging, particularly in microscopy, involves the precise identification and localization of each class of objects of interest based on their geometric shape. When images originate from diverse sources and multiple object classes need to be segmented simultaneously, this is referred to as multimodal and multi-class/label segmentation, respectively. Originally, this task relied on algorithms that utilized discontinuities or similarities in image properties to delineate objects. However, such methods face challenges in capturing and normalizing subtle and complex variations, especially when images are acquired through different microscopy modalities.

Deep learning, a subfield of machine learning where feature extraction is automated through self-learning processes, has revolutionized segmentation, particularly in the biomedical field. This approach outperforms traditional methods and opens up new possibilities. However, despite its promise, most deep learning models possess specific characteristics that limit their range of application.

Cellpose, the model employed by the Multimodal Imaging Centre (MIC), where my internship took place, falls within this context. This internship aims to assess the effectiveness of YOLOV8 for segmentation, with the objective of using it as an alternative and/or complementary solution to Cellpose. Should the results prove conclusive, YOLOV8 could be integrated into the Fiji image processing software, thereby providing end-users with similar accessibility as Cellpose.

## Recommanded Project folder structure

├── input/           # All input data is stored here.  
│   ├── train_images/  
│   │   ├── image01.png  
│   │   ├── image02.png  
│   │   └── ...  
│   ├── train_masks/        # All binary masks organized in respective sub-directories.  
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
│   │       └── ...  
│  
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


