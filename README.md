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
Dataset preprocessing was the biggest challenge, and took the largest amount of time. [explain more]

### 2. Training the model
[ explain training and limitations]


