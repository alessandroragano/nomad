# NOMAD Dataset 
NOMAD has been trained by finetuning the wav2vec 2.0 small model using degraded speech samples from Librispeech. We provide a script that allows you to degrade Librispeech or any other clean dataset. As degrading the whole ```train-clean-100``` partition of Librispeech will generate 225 GB of data, we do not recommend to do so. Instead, we directly provide a [link](https://zenodo.org/record/8380442/files/nomad_ls.tar.gz?download=1) to download the subset of the degraded Librispeech samples that we used to train and validate NOMAD. For the sake of completeness, we still provide instructions to generate the degraded samples as we did.

## Instructions
1. Download the datasets:
    *   [Librispeech](https://www.openslr.org/12) (clean data)
    *   [MS-SNSD](https://github.com/microsoft/MS-SNSD) (required for background noise degradation)

2. Modify two parameters in the YAML configuration file ```src/config/config_audio_degrader.yaml``` 
    * ```root``` path to Librispeech 
    * ```root_noise``` path to MS-SNSD
3. Run ```src/utils/audio_degrader_training.py```

This will generate: 

1. Degraded data in ```train-clean-100-degraded``` located where is Librispeech.
2. Converted wav files of clean Librispeech, which are needed to calculate the NSIM with ViSQOL (see below for more details).
3. A file ```degraded_data.csv``` in your working directory, including information on the generated data. 
4. A file ```degraded_data_visqol_format.csv``` in your working directory, which is formatted to run ViSQOL.

The script can be time-consuming, we recommend to run it in background e.g., using ```nohup```.

## Calculting NSIM offline
The NSIM is calculated using ViSQOL.
1. Follow the instructions in the [repo](https://github.com/google/visqol) to install ViSQOL
2. Set ```--batch_input_csv degraded_data_visqol_format.csv```

This will generate a csv file including patchwise NSIM scores. To train NOMAD we simply average them.

## Triplet Sampling
Triplet sampling can be done by running ```src/utils/nsim_triplet_sampling.py```. 
This script samples data from ```train_nsim.csv``` and ```valid_nsim.csv``` respectively and creates the triplet to train and validate NOMAD which are saved in```train.csv``` and ```valid.csv```.

Note that in the paper we have not used clean data in the triplets but background noise at 40 dB or OPUS/MP3 at 128 kbps which can be indistinguishable from clean data.
We have provided a modified script that also includes clean data together with the files ```train.csv``` and ```valid.csv``` which are the ones used to train NOMAD. 