# NOMAD: Non-Matching Audio Distance
_+++ **News:** We released SCOREQ, a new speech quality metric inspired by NOMAD and trained with MOS labels. SCOREQ can be used in no-reference, full-reference, and non-matching reference modes. [Link here](https://github.com/alessandroragano/scoreq)._

_While NOMAD is useful for training quality metrics without MOS, we recommend using SCOREQ for more accurate quality predictions._

## Description
NOMAD is a perceptual audio similarity metric trained without using human quality scores e.g., MOS. 

NOMAD embeddings can be used to:
* Measuring quality with any clean reference e.g., both paired and unpaired speech
* As a loss function to improve speech enhancement models

## Table of contents
- [NOMAD: Non-Matching Audio Distance](#description)
  * [Installation](#installation)
  * [Using NOMAD similarity score](#using-nomad-similarity-score)
    + [Using NOMAD from the command line](#using-nomad-from-the-command-line)
    + [Using NOMAD inside Python](#using-nomad-inside-python)
  * [Using NOMAD loss function](#using-nomad-loss-function)
    + [NOMAD loss weighting](#nomad-loss-weighting)
  * [Training](#training)
    + [Package dependencies](#package-dependencies)
    + [Dataset generation](#dataset-generation)
    + [Training the model](#training-the-model)
  * [Performance](#performance)
  * [ICASSP 2024](#icassp-2024)

## Installation
NOMAD is hosted on PyPi. It can be installed in your Python environment with the following command
```
pip install nomad_audio
```

The model works with 16 kHz sampling rate. If your wav files have different sampling rates, automatic downsampling or upsampling is performed.

NOMAD was built with Python 3.9.16.

## Using NOMAD similarity score
Data wav files can be passed in 2 modes:
* In ```mode 'dir'```, you need to indicate two directories for the reference and the degraded .wav files.  
* In ```mode 'csv```, you need to indicate two csv for the reference and the degraded .wav files.

Reference files can be any clean speech.

### Using NOMAD from the command line

To predict perceptual similarity of all .wav files between two directories:
```
python -m nomad_audio --mode dir --nmr /path/to/dir/non-matching-references --deg /path/to/dir/degraded
```

To predict perceptual similarity of all .wav files between two csv files:
```
python -m nomad_audio --mode csv --nmr /path/to/csv/non-matching-references.csv --deg /path/to/csv/degraded.csv
```

Both csv files should include a column ```filename``` with the absolute path for each wav file.


In both modes, the script will create two csv files in ```results-csv``` with date time format. 
* ```DD-MM-YYYY_hh-mm-ss_nomad_avg.csv``` includes the average NOMAD scores with respect to all the references in ```nmr_path``` 
* ```DD-MM-YYYY_hh-mm-ss_nomad_scores.csv``` includes pairwise scores between the degraded speech samples in ```test_path``` and the references in ```nmr_path```

Choosing where to save the csv files can be done by setting ```results_path```.

You can run this example using some .wav files that are provided in the repo:
```
python -m nomad_audio --mode dir --nmr_path data/nmr-data --test_path data/test-data
```

The resulting csv file ```DD-MM-YYYY_hh-mm-ss_nomad_avg.csv``` shows the mean computed using the 4 non-matching reference files:
```
Test File                  NOMAD
445-123860-0012_NOISE_15   1.587
6563-285357-0042_OPUS_64k  0.294
``` 

The other csv file ```DD-MM-YYYY_hh-mm-ss_nomad_scores.csv``` includes the pairwise scores between the degraded and the non-matching reference files:
```
Test File                  MJ60_10  FL67_01  FI53_04  MJ57_01
445-123860-0012_NOISE_15   1.627    1.534    1.629    1.561
6563-285357-0042_OPUS_64k  0.23     0.414    0.186    0.346
```

### Using NOMAD inside Python
NOMAD can be imported as a Python module. Here is an example:

```{python}
from nomad_audio import nomad 

nmr_path = 'data/nmr-data'
test_path = 'data/test-data'

nomad_avg_scores, nomad_scores = nomad.predict('dir', nmr_path, test_path)
```

## Using NOMAD loss function
NOMAD has been evaluated as a loss function to improve speech enhancement models. 

NOMAD loss can be used as a PyTorch loss function as follows:
```{python}
from nomad_audio import nomad 

# Here is your training loop where you calculate your loss
loss = mse_loss(estimate, clean) + weight * nomad.forward(estimate, clean)
```

We provide a full example on how to use NOMAD loss for speech enhancement using a wave U-Net architecture, see ```src/nomad_audio/nomad_loss_test.py```.
In this example we show that using NOMAD as an auxiliary loss you can get quality improvement:
* MSE -> PESQ = 2.39
* MSE + NOMAD loss -> PESQ = 2.60


Steps to reproduce this experiment:
* Download Valentini speech enhancement dataset [here](https://datashare.ed.ac.uk/handle/10283/2791)
* In ```src/nomad_audio/se_config.yaml``` change the following parameters
    * ```noisy_train_dir``` path to noisy_trainset_28spk_wav
    * ```clean_train_dir``` path to clean_trainset_28spk_wav
    * ```noisy_valid_dir``` path to noisy_validset_28spk_wav
    * ```clean_valid_dir``` path to clean_validset_28spk_wav
    * ```noisy_test_dir``` path to noisy_testset_wav
    * ```clean_test_dir``` path to clean_testset_wav

Notice that the Valentini dataset does not explicitly provide a validation partition. We created one by using speech samples from speakers ```p286``` and ```p287``` from the training set.

See the paper for more details on speech enhancement results using the model DEMUCS and evaluated with subjective listening tests.

### NOMAD loss weighting
We recommend to tune the weight of the NOMAD loss. Paper results with the DEMUCS model uses a weight of `0.1`. 
The U-Net model provided in this repo uses a weight equal to `0.001`.


## Training

### Package dependencies
After cloning the repo you can either pip install nomad_audio as above or install the required packages from ```requirements.txt```. If you install the pip package you will also have the additional nomad_audio module which is not needed to train NOMAD but only for usage.

### Dataset generation
NOMAD is trained on degraded samples from the Librispeech dataset.

[Download](https://zenodo.org/record/8380442/files/nomad_ls.tar.gz?download=1) the dataset to train the model.

In addition, we provide [instructions](data/nomad_dataset.md) to generate the dataset above. Notice that the process can be time-consuming, we recommend to download the dataset from the link.

### Training the model
The following steps are required to train the model:
1. Download wav2vec from this [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) and save it into ```pt-models```. You can skip this step if you installed the pip package with ```pip install nomad_audio``` in your working directory.
2. Change the following parameters in ```src/config/train_triplet.yaml```
    * ```root``` should be set to degraded Librispeech dataset path
3. From the working directory run: 
```
python main.py --config_file train_triplet.yaml
``` 

This will generate a path in your working directory ```out-models/train-triplet/dd-mm-yyyy_hh-mm-ss``` that includes the best model and the configuration parameters used to train this model.


## Performance
We evaluated NOMAD for ranking degradation intensity, speech quality assessment, and as a loss function for speech enhancement.
See the paper for more details. 
As clean non-matching references, we extracted 899 samples from the [TSP](https://www.mmsp.ece.mcgill.ca/Documents/Data/) speech database.

Here we show the scatter plot between NOMAD scores (computed with unpaired speech) and MOS quality labels. For each database we mapped NOMAD scores to MOS using a third order polynomial. 
Notice that performances are reported without mapping in the paper.

#### [Genspeech](https://arxiv.org/abs/2102.10449)
![genspeech](https://raw.githubusercontent.com/alessandroragano/nomad/main/figs/Genspeech_embeddings.png)

#### [P23 EXP1](https://www.itu.int/ITU-T/recommendations/rec.aspx?id=4415&lang=en)
![p23_exp1](https://raw.githubusercontent.com/alessandroragano/nomad/main/figs/P23_EXP1_embeddings.png)

#### [P23 EXP3](https://www.itu.int/ITU-T/recommendations/rec.aspx?id=4415&lang=en)
![p23_exp3](https://raw.githubusercontent.com/alessandroragano/nomad/main/figs/P23_EXP3_embeddings.png)

## ICASSP 2024
If you use NOMAD or the training corpus for your research, please cite our ICASSP 2024 [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10448028).
```
@INPROCEEDINGS{10448028,
  author={Ragano, Alessandro and Skoglund, Jan and Hines, Andrew},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment}, 
  year={2024},
  volume={},
  number={},
  pages={1011-1015},
  keywords={Degradation;Speech enhancement;Signal processing;Predictive models;Feature extraction;Acoustic measurements;Loss measurement;Perceptual measures of audio quality; objective and subjective quality assessment; speech enhancement},
  doi={10.1109/ICASSP48485.2024.10448028}}
``` 
[![DOI](https://zenodo.org/badge/681227455.svg)](https://doi.org/10.5281/zenodo.14735522)

The NOMAD code is licensed under MIT license.
Copyright © 2023 Alessandro Ragano



