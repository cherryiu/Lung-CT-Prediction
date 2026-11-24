# 3D CNN Lung Tumor Detection 
This tool is meant to predict potential tumors given lung CT/PET scans using a basic 3D convolutional neural network. Data was taken from the Cancer Imaging Archive and is loaded in from a .tcia file with corresponding xml annotations: 
https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/ 
</br>

## Usage
This CNN is specifically for preprocessing Cancer Imaging Archiev dataset linked above although, with a few modifications, it could possibly work on other similar datasets (folders with dicom images and annotations in a similar structure) </br>
After cloning the repo:
1. Create an anaconda environment:
`conda create -n [env_name] python=3.7`
2. Install requirements:
`pip install -r requirements.txt`
3. For files in .tcia format:
`python preprocess.py --dicom-path path/to/Lung-PET-CT-Dx --annotation-path path/to/Annotations`


## Workflow
This repo exists to document the learning experience that resulted from creating a 3D CNN from scratch with zero prior knowledge of designing neural networks and preprocessing medical imaging data. The following explains the workflow and pitfalls of this experience. 
</br>

### 1. Preprocessing

Dataset preprocessing was easily the biggest challenge and took up most of the project. The original dataset contains over 250,000 DICOM images, but those images still had to be grouped into scans, sorted, matched with their annotations, and converted into something the model could actually use.

The pipeline filters for CT series, groups slices by series ID, and sorts them using their Z-coordinate or Instance Number. Each slice is then matched with its XML annotation using its SOP Instance ID. From there, the slices are stacked into a 3D volume, normalized, and resized to a consistent 64 × 64 × 64 shape. Scans with missing metadata, corrupted files, or incompatible shapes are skipped instead of being forced through the pipeline.

This ended up being a good lesson in how messy real medical imaging data can be. Building the model was only one part of the project; getting clean and consistent data into it was most of the work.

### 2. Training the model

The model is a custom 3D CNN made up of convolution, max-pooling, normalization, and dropout layers. We used a 70/15/15 split for training, validation, and testing. Since the dataset is heavily imbalanced toward certain cancer types, we also experimented with class weights, smaller batch sizes, early stopping, learning-rate scheduling, and L2 regularization.

Most of the original code was written and tested in Google Colab before being moved to a campus VM. That move caused a surprising amount of extra work. We had to refactor several Colab-specific assumptions, containerize the project, work through TensorFlow/CUDA compatibility issues, and figure out why Docker was not actually using the GPU. To reduce the computational load, we lowered the input size from 128 × 128 × 128 to 64 × 64 × 64 and replaced a large `Flatten()` layer with global average pooling.

We also ran into a labeling bug that initially gave the model a suspicious 100% accuracy. After adding sanity checks, we found that every sample had accidentally been given the same label. Once that was fixed, the model reached 77.78% accuracy on a 50,000-image subset and 74.47% on the full dataset. The loss remained fairly high, so these results should be treated as a promising baseline rather than a finished diagnostic tool.

If we continued the project, the next steps would be data augmentation, a confusion matrix to better understand the misclassifications, and possibly a hybrid 2D/3D architecture that preserves spatial context without requiring quite as much compute.

