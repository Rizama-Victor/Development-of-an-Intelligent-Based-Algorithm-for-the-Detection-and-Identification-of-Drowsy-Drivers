# Development of an Intelligent Based Algorithm for the Detection and Identification of Drowsy 💤 Drivers 🚗

This repository contains the implementation of my research titled ""Development of an Intelligent-based Algorithm for the Detection and Identification of Drowsy Drivers: Preventive Mechanism for 
Road Accident.", published as a conference proceeding in the  21st International Conference and Exhibition on Power and Telecommunications (ICEPT 2025), Nigerian Institute of Electrical and Electronic Engineers (NIEEE), authored by Akinde O. K., Olaleye T. A., Ibitoye M. O., Taiwo S., Adetona M. O. & Rizama V.

---

## 🔍 Overview

Drowsiness remains a major cause of road accidents as many drivers fail to recognize fatigue early during long journeys. Although it may only last a few minutes, its potential consequences can be disastrous which includes the impairment of attention and alertness levels. This project focuses on the development of an Intelligent Driver Drowsiness Detection model that leverages computer vision and deep learning to monitor facial cues such as eye closure and yawning in real time. The system aims to accurately detect early signs of drowsiness which would be useful for alerting the driver before loss of control occurs, and ensuring safe driving under varying environmental conditions.

---

## 🧩 Research Objectives

- To curate a high-quality dataset of drowsy and none drowsy drivers.
- To train and develop an intelligent driver drowsiness detection model that is capable to detecting several drowsiness cues.
- To evaluate and assess the performance of the developed model in accurately detecting drowsy drivers in real-time.

---
  
## 🧰 Tools and Technologies Used

| **Tool/Libraries** | **Purpose in the Project**                                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python**         | Served as the primary programming language for implementing the drowsiness detection algorithm and training the deep learning model.           |
| **OpenCV**         | Used for image and video frame processing, including face and eye region detection, as well as feature extraction from real-time camera input. |
| **YOLOv8**         | Implemented for real-time detection of facial features and cues associated with driver drowsiness, such as eye closure and yawning.            |
| **Ultralytics**    | Provided the YOLOv8 framework used for model training, fine-tuning, and real-time inferencing on the drowsiness dataset.                       |
| **PyTorch**        | Served as the deep learning backend for building, training, and evaluating the neural network model.                                           |
| **TorchVision**    | Used to support model operations, dataset loading, and image transformations during training and testing phases.                               |
| **Google Colab**   | Offered GPU-accelerated cloud resources for training the deep learning model and conducting experiments efficiently.                           |
| **Roboflow**       | Utilized for dataset hosting, annotation, augmentation, and preprocessing to improve model training quality.                                   |
| **Display**        | Used to visualize training progress, test results, and detection outputs within the notebook environment.                                      |
| **Image**          | Enabled handling and representation of image data as Python objects for processing and display.                                                |
| **Shutil**         | Used for organizing and managing image and video files during dataset preparation and testing.                                                 |
| **CSV**            | Used to store and manage detection results, such as bounding box coordinates and classification outputs, in structured format.                 |
| **Codecs**         | Supported encoding and decoding of video data during the processing of real-time driver footage.                                               |

---

## 🧠 Model Building and Development

### Data Acquisition

A dataset comprising 18,140 images of drivers was collected, featuring different states of a driver such as `happy`, `Neutral`, `Eyes closed`, `Heavy Eyes`,  `Yawn` and `Bent Neck`. These images were sourced from online platforms such as Roboflow, Kaggle, Google Datasets, and YouTube. The class distribution of images by driver state is shown below:

| **Class**    | **Image Number** | 
|---------------|------------|
| Eye Closed    | 3299       | 
| Bent Neck     | 11         | 
| Heavy Eyes    | 3908       |
| Yawn          | 2886       |  
| Happy         | 111        |
| Neutral       | 7925       |   
| **Total**     |  **18140** |

### Data Cleaning and Structuring

To properly prepare the data for further use, the 18,140 images were adequately cleaned and sorted into different sections based on the condition of the driver by removal of blurred images or images of low resolution, irrelevant images that did not depict any drowsy symptoms etc. 

Furthermore, due to the minimal representation of the `Bent Neck` category (11 images), it was appropriately excluded from the dataset to avoid negatively affecting the dataset leaving the resulting amount to 18,129 images. The remaining data was re-organized into four symptom categories: `yawn`, `eyes closed`, and `heavy eyes` indicating drowsiness while `eyes opened` category predominantly represented non-drowsiness. This is because the initial `neutral` and `happy` sections were merged to collectively
represent non-drowsy symptoms. 
 
The inclusion of the non-drowsy symptoms i.e negative (-ve) samples was essential for improving the model’s robustness and overall performance. By integrating these negative categories, it made it possible for the model to differentiate between drowsy and non-drowsy drivers which improved its generalization capabilities and detection accuracy.

### 🏷️ Labelling and Annotation

Annotation was carried out using LabelImg, an open-source graphical tool for labeling images with bounding boxes in the YOLO format. The dataset, consisting of 18,129 images, was carefully annotated to capture facial cues related to drowsiness. Each time an object was labeled, the tool automatically generated a corresponding .txt file containing the bounding box parameters and class information.

Each annotated image included the following parameters:

**Class ID:** Represents the category of the detected facial feature (e.g., drowsy, non-drowsy, yawning, eyes closed).

**x:** X-coordinate of the bounding box’s top-left corner (horizontal start).

**y:** Y-coordinate of the bounding box’s top-left corner (vertical start).

**w:** Width of the bounding box in pixels.

**h:** Height of the bounding box in pixels.

These annotations provided the precise positional and categorical data required for the model to identify and localize drowsiness-related features within each image.

### Data Preparation

The specific data preparation technique used was class Balancing, implemented to maintain a balanced representation of all classes. Specifically, the number of instances for each class label was standardized to roughly 6,000 samples per class. This balance reduced the chances of the model being biased by ensuring an even distribution of weights across the different class categories (i.e yawn, eyes closed, eyes opened etc).

###  Data Pre-processing 
- **Auto-Orientation:** This ensured all images were displayed correctly regardless of their initial orientation during capture.
- **Image Resizing:** Resized Images to 640 X 640 pixels to align with YOLOv8 architecture for faster GPU training while retaining essential image details.

### Train-Test-Validation

Due to the large size of the dataset, the training, testing, and validation split was a ratio of 90:5:5 after preprocessing and augmentation.

### Model Training
