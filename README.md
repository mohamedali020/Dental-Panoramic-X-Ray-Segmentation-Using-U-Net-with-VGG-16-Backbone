# Dental X-ray Images Analysis Using Deep Learning(Segmentation Task..)🦷

##Abstract:

Radiographic examinations have a major role in assisting dentists to analyse the early teeth complications diagnosis such as infections, bone defects, and tumors. Unfortunately, relying only on the dentist’s opinion after 
a radiographic scan may lead to false-positive results, where it is proven that **3% of X-ray scan diagnoses are false resulting in psychological stress for the patients.** Researchers and doctors began using computer vision techniques to aid in diagnosing patients in the dentistry field because of the growing number of medical X-Ray images. In computer vision, various tasks are applied to digital images such as object detection, object tracking, and features recognition. **The most important computer vision technique is image segmentation, which is a deep learning technology used in the medical sector to detect key features in medical radiographs. Image segmentation works by dividing the pixels of an image into numerous segments, where each pixel is usually classified to belong to a specific class category in the image, this helps simplify the representation of the input image making the desired objects** 
easier to analyze by extracting the boundaries between objects to develop significant regions. There are numerous image segmentation algorithms with the goal to detect and extract the 
desired object from the image background. The two main types of image segmentation are semantic segmentation and instance segmentation where both techniques concatenate one another. **Semantic segmentation associates each pixel of the digital image with a class label** such as teeth in general, however, instance segmentation handles numerous objects of the same class independently.

##Challenges:

Image segmentation is **a difficult task for computers to execute**, the reason why it is not an easy challenge is due to the datasets used for segmentation tasks. Raw X-Ray images are usually corrupted with noise which may cause significant difficulties, to overcome some issues in the input images we **require numerous pre-processing techniques such as denoising** the images, 
scaling and normalizing the images, resizing all images, and adjusting images. Such **preprocessing techniques require high computational power and time** to load and process the data which may lead to runtime errors depending on the dataset used and the hardware handling the computation. Another challenge that is faced during image segmentation applications is the variability of the objects in the images, an example is the **dataset, teeth shapes vary between 
humans in terms of tooth location, tooth size, some images will include more teeth than others**, and teeth adhesion. This variability between input images makes it harder for the neural networks to learn and might **result in a large false-positive rate if not tackled correctly.**

## Project Overview📋

This is my first hands-on with image segmentation..In this project, **my goal is to develop an application with an AI-powered vision system that helps dentists diagnose panoramic dental X-rays. When a patient submits a panoramic dental X-ray, the dentist uploads it to the AI ​​system, which analyzes it to arrive at a final diagnosis. This improves diagnostic accuracy and speed, reduces errors, and leads to a better, more accurate treatment plan.** I will work on **a Segmentation Task**, which is very important, especially for medical images. I will also train the **U-net architecture Model and VGG-16-Backbone**..I will work on the **binary segmentation task**, which is its **role** in this project.Dividing the image into **two parts only:** A **mask is produced** from it containing the **Target**, which is the teeth with problems such as caries, impacted teeth, missing teeth, etc., depending on the **classes and annotations**, and the Background. **Each pixel in the image is classified as either 1 (Target) or 0 (Background).**

### Data Augmentation
To enhance the model's robustness and generalization, a significant portion of the 4772 training images was utilized. Data augmentation techniques were employed to increase dataset diversity, addressing the class imbalance. The following transformations were applied using the `Albumentations` library:

+Horizontal Flip: Flipping the images horizontally with 70% probability to simulate lateral perspectives.
+Rotate (limit=5): Rotating the images clockwise by up to 5 degrees with 70% probability to account for slight orientation variations.
+RandomBrightnessContrast: Adjusting brightness and contrast with limits of 0.1 and 30% probability to handle lighting variations.
+ShiftScaleRotate: Shifting, scaling, and rotating with limits of 0.1, 10 degrees, and 50% probability to simulate positional and scaling differences.

## Dataset Details
- **Source:** Available on Kaggle (https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task) and originally sourced from Roboflow (https://universe.roboflow.com/arshs-workspace-radio/vzrad2).
- **Description:** This dataset fuels AI innovation in tooth segmentation for improved dental care. It includes panoramic X-ray images divided into binary and multi-class segmentation tasks, with three annotation files for training, validation, and testing.
- **Composition:**
  - **Train Images:** 4772
  - **Train Mask Labels:** 4772
  - **Validation Images:** 2071
  - **Validation Mask Labels:** 2071
  - **Test Images:** 1345
  - **Test Mask Labels:** 1345
- **Image Size:** Primarily (640, 640) pixels (Width, Height), with 10 images confirmed at this resolution; all other images are uniformly sized.
- **Unique Classes in Annotations:** {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14}
![Sample images from dataset showing image segmentation output](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/0d220dea-Farasati_Simin_35yo_08062021_145847_jpg.rf.478a679c3667801fa26068e518dea362.jpg)
![Sample images from dataset showing Mask segmentation output](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/00cf39c1-Karaptiyan_Robert_50yo_13032021_185908_jpg.rf.98b2e72cb9a26e75d40df97e04473ada.jpg_mask.png)

## Methodology

the methodology used in building the segmentation models will be discussed 
such as: 
1- Data Collection 
2- Data pre-processing 
3- Deep Learning segmentation model architectures.  
4- Hyper-tuning and evaluation metrics.

### Model Architecture

- **U-Net:** A widely adopted architecture for semantic segmentation, initially designed for medical images. It excels in tasks like brain tumor, lung, and cell segmentation, and is adapted here for dental X-rays.
- **Backbone:** VGG16 provides pre-trained feature extraction, enhancing model performance.



## Results
- **Performance:** After 8 epochs, the model achieves a  accuracy: 0.9859 (training) and 0.9878 (validation) and Dice Coefficient of 0.71 (training), 0.7123 (validation)
![Screenshot Example](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/Screenshot_15.png)

## Final Output:
![WhatsApp Image Example](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/WhatsApp%20Image%202025-06-17%20at%2017.56.19_65fbd25a.jpg)

## Future Work and Developments
The future work that could be developed in the field of image segmentation includes **reducing the computational time and power.**  Smart Imaging is ongoing research in the fields of image segmentation, it is known that segmentation algorithms are **able to perform well on images with high quality after applying pre-processing on the images**, unfortunately, it is proven that **those algorithms fail on lowquality images**. **Smart imaging is the process of automatically improving the quality of the input images**, which would be **very effective to increase the accuracy of segmenting medical** images. It is stated that it uses a deep learning approach **to effectively improve the image quality and denoise it automatically**, the process is also being tested on **improving image resolution**, motion correction, shadow detection, and denoising, and **Data Expansion and Multi-class Integration with Mask R-CNN**


# Dental X-ray Images Analysis Using Deep Learning (Segmentation Task) 🦷

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![GitHub stars](https://img.shields.io/github/stars/USERNAME/REPO.svg)](https://github.com/USERNAME/REPO/stargazers) [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=fff)](https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task)

![X-ray Example](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/1-s2.0-S0010482522010046-ga1.jpg)

## 📑 Table of Contents
- [Abstract](#abstract)  
- [Challenges](#challenges)  
- [Project Overview](#project-overview)  
- [Dataset Details](#dataset-details)  
- [Methodology](#methodology)  
- [Model Architecture](#model-architecture)  
- [Results](#results)  
- [Final Output](#final-output)  
- [Future Work](#future-work)  
- [Installation & Usage](#installation--usage)  

---

## 🧾 Abstract

**Radiographic examinations** have a major role in assisting **dentists to analyse the early teeth complications diagnosis** such as infections, bone defects, and tumors. Unfortunately, relying only on the dentist’s opinion after 
a radiographic scan may lead to false-positive results, where it is proven that **3% of X-ray scan diagnoses are false resulting in psychological stress for the patients.** Researchers and doctors began using computer vision techniques to aid in diagnosing patients in the dentistry field because of the growing number of medical X-Ray images. In computer vision, various tasks are applied to digital images such as object detection, object tracking, and features recognition. **The most important computer vision technique is image segmentation, which is a **deep learning technology used in the medical sector to detect key features in medical radiographs**. Image segmentation works by dividing the **pixels of an image into numerous segments**, where each pixel is usually classified to belong to a specific class category in the image, this helps simplify the representation of the input image making the desired objects** 
easier to analyze by extracting the boundaries between objects to develop significant regions. There are numerous image segmentation algorithms with the goal to detect and extract the 
desired object from the image background. The **two main types of image segmentation are semantic segmentation and instance segmentation** where both techniques concatenate one another. **Semantic segmentation associates each pixel of the digital image with a class label** such as teeth in general, however, instance segmentation handles numerous objects of the same class independently.

---

## ⚠️ Challenges

- X-rays often have **noise**, requiring denoising, resizing, normalization, and scaling.
- **High variability** in tooth shape, size, and alignment across patients can lead to false positives.
- **Preprocessing** demands heavy computation and can cause runtime issues depending on available hardware.
- **Class imbalance**: background pixels heavily outnumber target pixels, requiring careful training strategy.

---

## 📋 Project Overview

- I built an **AI-powered system** that helps dentists analyze panoramic X-rays.
- They upload an X-ray → the model segments problem areas (e.g. caries, impacted or missing teeth).
- A **U-Net-based binary segmentation** identifies dental regions (target vs. background).
- I trained a **U-Net with VGG-16 encoder**, optimized to classify each pixel correctly.

---
 
## 📊 Dataset Details

- **Source:** Kaggle ([link](https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task)), originally from Roboflow.
- **Counts & Splits:**
  - Train: 4,772 images + masks  
  - Validation: 2,071 images + masks  
  - Test: 1,345 images + masks  
- **Resolution:** Mostly 640×640 px  
- **Classes:** {1,2,3,…,14} annotation IDs in masks  
- **Samples:**  
  ![Dataset Input Sample](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/0d220dea-Farasati_Simin_35yo_08062021_145847_jpg.rf.478a679c3667801fa26068e518dea362.jpg)  
  ![Dataset Mask Sample](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/00cf39c1-Karaptiyan_Robert_50yo_13032021_185908_jpg.rf.98b2e72cb9a26e75d40df97e04473ada.jpg_mask.png)

---

## ⚙️ Methodology

1. **Data Collection** → Gather and inspect the dataset.  
2. **Pre‑processing** → Resize, normalize, and ensure consistent formatting.  
3. **Data Augmentation** (via Albumentations):
   - Horizontal flip (70%)  
   - Rotation ≤ 5° (70%)  
   - Random brightness/contrast (limit 0.1, 30%)  
   - Shift/Scale/Rotate (10°, 50%)  
4. **Segmentation Model** → Build U-Net with VGG‑16 backbone.  
5. **Hyper‑tuning & Metrics** → Use Dice loss; track accuracy & Dice coefficient.

---

## 🧱 Model Architecture

- **U‑Net**: Encoder–decoder with skip connections, ideal for medical imaging.  
- **Encoder:** VGG‑16 (ImageNet pretrained) for robust feature extraction.  
- **Decoder:** Upsampling layers with skip‑connections to restore spatial details.  


---

## 📈 Results

After 8 epochs:

| Metric            | Training | Validation |
|------------------|----------|------------|
| Accuracy         | 0.9859   | 0.9878     |
| Dice Coefficient | 0.71     | 0.7123     |

![Training Curves](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/Screenshot_15.png)

---

## 🎯 Final Output

A binary mask highlighting target tooth regions.

![WhatsApp Image Example](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/WhatsApp%20Image%202025-06-17%20at%2017.56.19_65fbd25a.jpg)

---

## 🔭 Future Work & Developments

- **Speed Optimizations:** Reduce computational cost via model trimming or quantization.  
- **Smart Imaging:** Automatically enhance low-quality X-rays (denoising, resolution upscaling).  
- **Multi-class Segmentation:** Use Mask R‑CNN or multiclass U-Net for identifying tooth types.  
- **Motion/Shadow Correction:** Handle real-world X-ray artifacts.  
- **Clinical Integration:** Incorporate dentist feedback into model refinement.

---


## 🛠️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/USERNAME/REPO.git
cd REPO

# Install dependencies (Python 3.7+)
pip install -r requirements.txt




