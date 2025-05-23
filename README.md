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

This repository contains the code for a violin vs. viola audio classification pipeline. It combines two datasets—the Solos dataset (scraped and extended) and the URMP dataset (violin and viola WAV files)—to train three machine learning model that can distinguish between violin and viola recordings.

The pipeline includes data download, segmentation, feature extraction, model training, evaluation, and a upload-based classification demo in Google Colab.

Below are the instructions to install all dependencies and run the project

---

## Installation

All done through Colab:
```
!git clone https://github.com/JooshJin/ViolinViolaClassifier.git
%cd ViolinViolaClassifier

# Install dependencies
!pip install yt-dlp librosa scikit-learn matplotlib
```
### Check that versions are accurate:
```
Python version: 3.11.12
numpy version: 2.0.2
librosa version: 0.11.0
scikit-learn version: 1.6.1
matplotlib version: 3.10.0
pandas version: 2.2.2
tqdm version: 4.67.1
joblib version: 1.5.0
```


---

## Data Preparation

### **DOWNLOAD AND RUN EVERYTHING IN A NON-YALE GOOGLE DRIVE AND COLAB IF POSSIBLE**
#### Some issues occur especially during the Solos Dataset download because of Yale's security, and cookies extension don't work on Yale Google accounts.

### 1. URMP Dataset (Violin & Viola recordings)

link to folders in drive (only when downloading from github, not applicable for class submission): [https://drive.google.com/drive/folders/1yqhQIs22p_8s-hsgIW2-WbmEn3FSZfF7?usp=sharing](https://drive.google.com/drive/folders/1yqhQIs22p_8s-hsgIW2-WbmEn3FSZfF7?usp=sharing)

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

### 2. Solos Dataset (YouTube recordings)
1. Ensure you are logged into YouTube (needed for manifest download,  or else you get the error message).
```
ERROR: [youtube] JrNn2Ns7k-g: Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies for the authentication. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  for how to manually pass cookies. Also see  https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies  for tips on effectively exporting YouTube cookies
```
2. Use the provided manifest loader to fetch audio URLs and metadata from YouTube.
3. The pipeline script will download, segment, and process the Solos recordings.

---

## Usage

### Google Colab

1. Open `VVclassifier_main_pipeline.ipynb` in Colab. 

Can be anywhere in your drive, preferably a non-Yale or other high-security Google account, as that can cause issues with retrieving the solos dataset on Youtube. See [Solos Dataset](#2.-Solos-Dataset)

2. Run the cells in order, ensuring you mount Drive and log into YouTube beforehand.

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

