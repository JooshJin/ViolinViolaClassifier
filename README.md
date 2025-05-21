# ViolinViolaClassifier

**Yale University CPSC 381 Final Project**
**Author:** Joshua Jin

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Usage](#usage)
5. [Version Info](#version-info)
6. [Credits](#credits)

---

## Overview

This repository contains the code for a violin vs. viola audio classification pipeline. It combines two datasets—the Solos dataset (scraped and extended) and the URMP dataset (violin and viola WAV files)—to train a machine learning model that can distinguish between violin and viola recordings.

The pipeline includes data download, segmentation, feature extraction, model training, evaluation, and upload-based classification in Google Colab.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/ViolinViolaClassifier.git
   cd ViolinViolaClassifier
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

---

## Data Preparation

### 1) URMP Dataset (Violin & Viola recordings)

1. Unzip the ViolinViolaData folder (contains URMP dataset) and store in Google Drive under:

   ```text
   /content/drive/My Drive/ViolinViolaData/Violin-URMP/
   /content/drive/My Drive/ViolinViolaData/Viola-URMP/
   ```
2. In Google Colab, mount your Drive and copy files (already done in the main notebook):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !mkdir -p data/raw/Violin_URMP data/raw/Viola_URMP
   !cp -r "/content/drive/My Drive/ViolinViolaData/Violin-URMP/"* data/raw/Violin_URMP/
   !cp -r "/content/drive/My Drive/ViolinViolaData/Viola-URMP/"*  data/raw/Viola_URMP/
   ```

### 2) Solos Dataset (YouTube recordings)

1. Ensure you are logged into YouTube (needed for manifest download).
2. Use the provided manifest loader to fetch audio URLs and metadata from YouTube.
3. The pipeline script will download, segment, and process the Solos recordings.

---

## Usage

### Google Colab

1. Open `vv_classifier_main_pipeline.ipynb` in Colab.
2. Run the cells in order, ensuring you mount Drive and log into YouTube when prompted.

---

## Version Info

```text
Python:       3.11.12
numpy:        2.0.2
librosa:      0.11.0
scikit-learn: 1.6.1
matplotlib:   3.10.0
pandas:       2.2.2
tqdm:         4.67.1
joblib:       1.5.0
```

---

## Credits

* **Solos Dataset**
  Original JSON manifests: [https://github.com/JuanFMontesinos/Solos/tree/master/Solos/json\_files](https://github.com/JuanFMontesinos/Solos/tree/master/Solos/json_files)

  * Scraped for usable YouTube URLs and extended with additional links by the author.

* **URMP Dataset**
  University of Rochester Multi-Modal Music Performance (URMP) Dataset: [https://labsites.rochester.edu/air/projects/URMP.html](https://labsites.rochester.edu/air/projects/URMP.html)

  * Used exclusively the violin and viola WAV files for this project.

---

**Enjoy experimenting with violin vs. viola classification!**

