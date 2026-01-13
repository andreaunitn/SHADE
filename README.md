# SHADE: Sharing Device Identification on Images

<div align="center">
  <img src="figs/shade_visual_abstract.pdf" alt="SHADE Dataset Overview" width="60%">
</div>

<br>

**SHADE** (SHAring DEvice identification) is a dataset designed to investigate the forensic traces left by different devices and software interfaces when sharing images on social media platforms (specifically **WhatsApp**).

This repository contains the official implementation and data description for the paper: **"Sharing Device Identification on Images from Social Media Platforms"**.

__Paper__: ["Sharing Device Identification on Images from Social Media Platforms"](https://ieeexplore.ieee.org/abstract/document/9948824?casa_token=U49IALImPg4AAAAA:6ZRahwSJyDGnouXWqlbqBJmJwiTei7mmrj-VhAdoXPttO9sNFrwuIPxQJd3GHyVjxmcs9SYb)
</br>

## 📄 Abstract

In real-world scenarios, users upload images to social platforms using multiple methods: via smartphone apps, desktop applications, or web browsers. Being able to detect these different sharing modalities represents a valuable insight for forensic purposes. 

**SHADE** is the first collection of real-world images shared from different devices, operating systems, and user interfaces. We provide this asset to the forensic community to investigate the peculiar artifacts introduced by different sharing modes and to validate detection algorithms.

## 📊 Dataset Overview

The dataset is built starting from **900** source images (derived from the RAISE dataset), which are then shared on WhatsApp through **7 different modalities**.

*   **Total Images**: 7,200
    *   6,300 Shared Images
    *   900 Non-shared (Originals)
*   **Platform**: WhatsApp
*   **Source Resolution**: Resized to 3 resolutions ($337 \times 600$, $1012 \times 1800$, $1687 \times 3000$).
*   **Compression**: Pre-compressed with 6 Quality Factors (QF = 50, 60, 70, 80, 90, 100).

### Sharing Classes (Labels)

The images are categorized by the specific hardware/software combination used for the upload:

| Label | Description |
| :--- | :--- |
| `ANDROID` | Mobile application for Android |
| `IPHONE` | Mobile application for iOS |
| `APP-MAC` | Desktop application for macOS |
| `APP-WIN` | Desktop application for Windows 10 |
| `WEB-IPAD` | Browser (Safari) for iPadOS |
| `WEB-MAC` | Browser (Safari) for macOS |
| `WEB-WIN` | Browser (Chrome) for Windows 10 |

## 📂 Directory Structure

The dataset is organized hierarchically. Images are first divided by **Sharing Mode**, then by **Quality Factor (QF)**.

```text
SHADE/
├── original/
├── ANDROID/
│   ├── QF-50/
│   ├── ...
│   └── QF-100/
├── IPHONE/
├── APP-MAC/
├── APP-WIN/
├── WEB-IPAD/
├── WEB-MAC/
└── WEB-WIN/
```

### Naming Convention
Files follow a strict naming structure containing the ID, height, and width:
```text
original-[id]-[h]x[w].jpeg
```

## 📥 Download
The dataset is publicly available. You can download it from the MMLab website:

[**Download SHADE Dataset**](https://mmlab.disi.unitn.it/resources/published-datasets)

## 🧪 Methodology & Experiments

The paper validates the dataset by extracting heterogeneous feature descriptors:
1.  **DCT Coefficients**: Histograms of AC frequencies.
2.  **Metadata**: Analysis of EXIF availability and structure.
3.  **JPEG Header**: Quantization tables and Huffman encoding.

We demonstrated that different uploading interfaces (e.g., *Desktop App* vs. *Web Browser*) can be successfully recognized using classifiers like **Random Forest** and **SVM**, with the `IPHONE` class showing 100% separability.

## ✏️ Citation

If you use this dataset or code in your research, please cite the following paper:

```bibtex
@inproceedings{tomasoni2022sharing,
  title={Sharing device identification on images from social media platforms},
  author={Tomasoni, Andrea and Verde, Sebastiano and Boato, Giulia},
  booktitle={2022 IEEE 24th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}
```
## 👨‍💻 Authors & Contact

*   **Andrea Tomasoni** - University of Trento - [GitHub](https://github.com/andreaunitn)
*   **Sebastiano Verde** - University of Trento
*   **Giulia Boato** - University of Trento

For any questions regarding the dataset or the implementation, please open an issue in this repository.
