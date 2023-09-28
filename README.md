# Non-Matching Audio Distance (NOMAD)

NOMAD is a differentiable perceptual similarity metric that measures the distance of a degraded signal against non-matching references (unpaired speech).
The proposed method is based on learning deep feature embeddings via a triplet loss guided by the Neurogram Similarity Index Measure (NSIM) to capture degradation intensity. During inference, the similarity score between any two audio samples is computed through Euclidean distance of their embedding.
NOMAD can be also used as a loss function to improve speech enhancement models.

## Table of contents
- [Non-Matching Audio Distance (NOMAD)](#non-matching-audio-distance--nomad-)
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
  * [Paper and license](#paper-and-license)

## Installation
NOMAD is hosted on PyPi. It can be installed in your Python environment with the following command
```
pip install nomad_audio
```

## Using NOMAD similarity score
### Using NOMAD from the command line
NOMAD similarity score can be used to measure perceptual similarity between any two signals. NOMAD can be used with unpaired speech i.e., any clean speech can serve as a reference. You can use NOMAD from the command line as follows:  

```
python -m nomad_audio --nmr_path /path/to/dir/non-matching-references --test_path /path/to/dir/degraded
```

The script creates two csv files in ```results-csv``` with date time format. 
* ```DD-MM-YYYY_hh-mm-ss_nomad_avg.csv``` includes the average NOMAD scores with respect to all the references in ```nmr_path``` 
* ```DD-MM-YYYY_hh-mm-ss_nomad_scores.csv``` includes pairwise scores between the degraded speech samples in ```test_path``` and the references in ```nmr_path```

You can choose where to save the csv files by setting ```results_path```. 

### Using NOMAD inside Python
You can import NOMAD as a module in Python. Here is an example:

```{python}
from nomad_audio import nomad 

nmr_path = 'data/nmr-data'
test_path = 'data/test-data'

nomad_avg_scores, nomad_scores = nomad.predict(nmr_path, test_path)
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

See the paper for more details on speech enhancement results using the model DEMUCS.

### NOMAD loss weighting
We recommend to tune the weight of the NOMAD loss. Paper results with the DEMUCS model has been done by setting the weight to `0.1`. 
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
1. Download wav2vec from this [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) and save it into ```pt-models```. If you ran above ```pip install nomad_audio``` in your working directory you can skip this step.
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

Here we show the scatter plot between NOMAD scores and MOS quality labels. For each database we mapped NOMAD scores to MOS using a third order polynomial. 
Notice that performances are reported without mapping in the paper.

#### [Genspeech](https://arxiv.org/abs/2102.10449)
![genspeech](https://github.com/alessandroragano/nomad/blob/main/figs/Genspeech_embeddings.png?raw=true)

#### [P23 EXP1](https://www.itu.int/ITU-T/recommendations/rec.aspx?id=4415&lang=en)
![p23_exp1](https://github.com/alessandroragano/nomad/blob/main/figs/P23_EXP1_embeddings.png)

#### [P23 EXP3](https://www.itu.int/ITU-T/recommendations/rec.aspx?id=4415&lang=en)
![p23_exp3](https://github.com/alessandroragano/nomad/blob/main/figs/P23_EXP3_embeddings.png)

## Paper and license
Pre-print will be available soon.
The NOMAD code is licensed under MIT license.