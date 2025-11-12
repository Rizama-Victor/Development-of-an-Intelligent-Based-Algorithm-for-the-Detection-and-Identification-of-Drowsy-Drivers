# Development of an Intelligent Based Algorithm for the Detection and Identification of Drowsy 💤 Drivers 🚗

This repository contains the implementation of my research titled _"Development of an Intelligent-based Algorithm for the Detection and Identification of Drowsy Drivers: Preventive Mechanism for Road Accident."_, published as a conference proceeding in the  21st International Conference and Exhibition on Power and Telecommunications (ICEPT 2025), Nigerian Institute of Electrical and Electronic Engineers (NIEEE), authored by Akinde O. K., Olaleye T. A., Ibitoye M. O., Taiwo S., Adetona M. O. & Rizama V.

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
| **Roboflow**       | Utilized for dataset augmentation, and preprocessing to improve model training quality.                                   |
| **LabelImg**       | Utilized for dataset annotation/labelling.                                   |
| **Google Drive**       | Used for hosting the dataset.                                   |
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

To properly structure the data for further use, the 18,140 images were adequately cleaned and sorted into different sections based on the condition of the driver by removal of blurred images or images of low resolution, irrelevant images that did not depict any drowsy symptoms etc. 

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

###  Data Pre-processing and Augnentation 

#### Pereprocessing Techniques

- **Auto-Orientation:** This ensured all images were displayed correctly regardless of their initial orientation during capture.
- **Image Resizing:** Resized Images to 640 X 640 pixels to align with YOLOv8 architecture for faster GPU training while retaining essential image details.

#### Augmentation Techniques 

- **Horizontal Flips:** Flipped Images along the vertical axis to create mirrowed versions to enable the model recognize drowsy drivers regardless of their left-to-right orientation.

**Note:** The augmentation process resulted in an increase in the number of images from 18,129 to 20,927 images. The augmented dataset was then randomly shuffled and splitted into training, test and validation sets to help improve generalization, reduce bias and improve the overall performance.

### Train-Validation-Test Split

Due to the large size of the dataset, the training, testing, and validation split was a ratio of 90:5:5 after preprocessing and augmentation. The training set (18,835 images) served as the largest subset and was used to train the YOLOv8 model, enabling it to learn essential patterns, features, and relationships between inputs and labels. The validation set (1046 images) was used during training to monitor performance on unseen data, fine-tune hyperparameters, and prevent overfitting. Finally, the testing set (1046 images) was reserved for the final model evaluation, providing an unbiased measure of its accuracy, robustness, and generalization to real-world scenarios.

### Model Training

The training procedure for the model involved mounting the drive in the Google colab virtual environment, installing the ultralytics library, importing YOLO, importing the dataset from the google drive, and finally training the model. The training time lasted for a total of  8.805 hours.

### 🤖 Model Summary

| Hyperparameter | Value |
|------------------------|---------------------------|
| Number of Epochs | 100 |
| Learning Rate | 0.01 |
| Image Input Size | 640 x 640 |
| Total Number of Classes | 6 |
| Batch Size | 16 |
| Activation Function | SiLU |
| Momentum | 0.937 |
| IoU Threshold | 0.7 |
| Optimizer | auto (AdamW at initial layers for early convergence and SGD at final layers for fine tuning) |

### Evaluation Metrics

| **Metric**      | **Value** | **Remarks**                                                                                                          |
| --------------- | --------- | -------------------------------------------------------------------------------------------------------------------- |
| **Precision**   | 0.895     | The model accurately identified most drowsiness cues (e.g., eye closure, yawning) with few false detections.         |
| **Recall**      | 0.914     | Demonstrated high sensitivity, successfully detecting the majority of drowsy facial cues across test samples.        |
| **mAP@50**      | 0.952     | Showed strong detection accuracy at a relaxed IoU threshold (50%), indicating reliable recognition of drowsy signs.  |
| **mAP@50-95**   | 0.763     | Reflected consistent performance across varying IoU thresholds, showing robust generalization to unseen faces.       |
| **F1-Score**    | 0.900     | Balanced trade-off between precision and recall, suggesting dependable overall drowsiness detection capability.      |
| **Box Loss**    | 0.732     | Indicated moderate localization error, showing the model could further refine bounding box placement on facial cues. |
| **Object Loss** | 0.400     | Demonstrated effective differentiation of drowsy and non-drowsy symptoms, with minimal classification errors.        |

### 🎥 Demo 

<p align="center">
    <img src="Demo/demo.gif" alt="A short video clip of the Model's Performance on a Test Video" width="1500"/>
    <br>
    <em> A Snip Video Clip of the Model's Performance on a Test Video</em>
</p>

**Note:** For the full test video showing detections for all classes, kindly [chick here](https://stfutminnaedung-my.sharepoint.com/:f:/g/personal/victor_m1901621_st_futminna_edu_ng/Er9QjNH2vntDjOi3TBZlHqQBt31NZODcDm0YpgTE7y7byA?e=d1nciu).

### Result Plots
<p align="center">
    <img src="results.png" alt="The Model's Result Summary" width="1500"/>
    <br>
    <em> Fig 2: The Model's Result Summary</em>
</p>

<p align="center">
    <img src="confusion_matrix.png" alt="The Confusion matrix" width="1500"/>
    <br>
    <em> Fig 3: Confusion Matrix</em>
</p>

## 💡 Key Insights

- The research addressed the key limitations of existing driver drowsiness detection systems, which typically only determined whether a driver was drowsy or not without identifying the specific symptoms responsible for that classification. It achieved this by classifying drowsy symptoms such as `yawn`, `eyes closed`, and `heavy eyes`, and not drowsy symptoms such as `eyes open` in addition to the overall `drowsy` and `not drowsy` detections.
- The model demonstrated reliable detection performance under varied lighting conditions (both bright and dark) and from multiple viewing angles, including front, left, right, and rear views of the driver.

---

## 🔮Future Work
Future work could focus on combining the model with other sensor-based modalities such as eye-tracking, head pose estimation, or steering behavior analysis to create a multi-modal driver monitoring system. This hybrid approach would improve detection accuracy by validating visual cues (like yawning or eye closure) with physiological or behavioral signals, making the system more adept to lighting conditions, camera angles, and driver variability.

---

## 📌 Note

Please kindly note that this README file is a summarized version of the full implementation of this research. The complete implementation can be accessed via the [program script](Driver_Drowsiness_Detection_System_Using_YOLO.ipynb). Dataset and Model Weights can be provided upon request.

---
