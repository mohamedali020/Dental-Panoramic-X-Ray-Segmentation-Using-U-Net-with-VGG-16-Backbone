# Dental X-ray Binary Segmentation with U-Net and VGG16 🦷

## Project Description
This project implements a deep learning-based solution for binary semantic segmentation of Dental Panoramic X-ray images, aiming to assist dentists in diagnosing dental conditions such as caries, impacted teeth, and missing teeth. The AI-powered vision system processes 256x256 RGB X-ray images, generating pixel-wise binary masks (1 for Target dental structures, 0 for Background) to enhance diagnostic accuracy, reduce errors, and optimize treatment planning. Built as my first hands-on experience with image segmentation, this work leverages the U-Net architecture with a pre-trained VGG16 backbone, trained on a Kaggle dataset.

## Dataset Details
- **Source:** Kaggle Dataset (ID: 7644979).
- **Size:** 4772 training images and 2071 validation images.
- **Format:** 256x256x3 RGB images with corresponding binary masks.
- **Challenge:** Significant class imbalance (~3.79% Target pixels), addressed using data augmentation (e.g., HorizontalFlip, Rotate, GaussNoise) to improve the Target-to-Background ratio (1:38 post-augmentation).

## Methodology
### Image Segmentation
Image segmentation groups pixels with similar attributes into meaningful regions. In this binary segmentation task, each pixel is labeled as either Target (dental structures of interest) or Background, enabling detailed pixel-level analysis critical for medical diagnostics.

### Model Architecture
- **U-Net:** A widely adopted architecture for semantic segmentation, initially designed for medical images. It excels in tasks like brain tumor, lung, and cell segmentation, and is adapted here for dental X-rays.
- **Backbone:** VGG16 provides pre-trained feature extraction, enhancing model performance.
- **Training Setup:** TensorFlow/Keras with a batch size of 4, 12 epochs (targeting 50), and callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau). Custom Dice Loss and Dice Coefficient are used as loss function and metric, respectively.

### Implementation Steps
1. **Data Preprocessing:** Resize images to 256x256, normalize pixel values, and apply augmentation.
2. **Model Training:** Optimize using GPU acceleration on Kaggle.
3. **Visualization:** Generate overlaid masks, standalone masks, and Grad-CAM heatmaps to interpret model focus.

## Results
- **Performance:** After 12 epochs, the model achieves a Dice Coefficient of 0.6068 (training) and 0.6148 (validation), with losses reducing to 0.9659 and 0.9993, respectively.
- **Classification Metrics:** 99% accuracy, 74% precision, 66% recall, and 70% F1-score for the Target class (support: 2,082,218 pixels), indicating robust background segmentation but room for Target improvement.
- **Visual Outputs:** Masks and heatmaps provide qualitative insights into segmentation accuracy.

## Usage Instructions
1. Clone the repository: `git clone <repository-url>`.
2. Install dependencies: `pip install -r requirements.txt` (to be updated with TensorFlow, Keras, OpenCV, etc.).
3. Open and run the notebook: `dental-x-ray-binary-segmentation-u-net-vgg16.ipynb` in Jupyter with GPU support.
4. Upload the Kaggle dataset (ID: 7644979) and adjust paths as needed.

## Future Work and Developments
- **Computational Efficiency:** Reduce training time and resource demands for scalability.
- **Smart Imaging:** Employ deep learning to automatically enhance low-quality images through denoising, resolution improvement, motion correction, and shadow detection, boosting segmentation accuracy for medical images.
- **Advanced Segmentation:** Expand the dataset and transition to multi-class segmentation using Mask R-CNN to classify diverse dental structures (e.g., caries, bone loss).

## Acknowledgments
Gratitude to the Kaggle community for providing the dataset and resources, and to the open-source community for U-Net and VGG16 implementations.
