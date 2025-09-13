# Representational Learning for Automated Segmentation And Registration With Application To Multiparametric Kidney MRI

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
<a href="https://www.tensorflow.org/"> <img src="https://img.shields.io/badge/TensorFlow-2.8-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

📄 *This work is currently under review.*

---

## About the Project

This project aims to enhance the segmentation and registration of multiparametric renal MR images by leveraging domain-specific contrast information from unlabeled images. 
We utilize contrastive learning to develop a tissue-specific representation from multi-contrast images, capturing distinctive features for improved segmentation. 
The learned representations also enable robust image registration across varying contrasts, creating a comprehensive framework for feature extraction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lab-midas/Renal_fMRI.git

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

## Acknowledgements

This project was developed with the help of the following repository:

- [Constrained Contrastive Learning](https://github.com/lunastra26/multi-contrast-contrastive-learning) – for pre-training contrastive learning technique.  
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph) – for the motion estimation losses and metrics.
