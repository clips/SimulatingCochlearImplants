Code for obtaining results described in the following paper:

> Grimm R., Michèle P., Gillis S. and Daelemans W. Simulating speech processing with cochlear implants: How does channel interaction
affect learning in neural networks? Submitted to PLOS One.

## OS and Dependencies

This project is written in Python (version 3.6.6) and R (version 3.3.3), both on Ubuntu 14.04. The biggest part of the code is written in Python, and a small part for statistical analysis is written in R. 
The Python component requires the following packages (the version we used is given in parentheses):
> Keras (2.2.2)  
numpy (1.15.0)  
scipy (1.1.0)  
tqdm (4.24.0)  
tensorflow (1.9.0)  
librosa (0.6.2)  
matplotlib (2.2.2)  
scikit_learn (0.19.2)  

The R scripts, which we use to plot and inspect the results, rely on the following packages: 
> DBI (0.7)  
boot (1.3-18)  
ppcor (1.1)  
ggplot2 (2.2.1)  
extrafont (0.17)  

## Get the Speech Data

In order to run the experiments, you need to download the freely available 
[Google speech command data set](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) (Note that this data set might be extended at some point in the future, and that we obtained our copy in August 2018). You also need to purchase and download the
[TIMIT corpus](https://catalog.ldc.upenn.edu/LDC93S1).  

In the project directory, create a new folder named __GSC__ and unpack the speech command data set to this folder (so that the WAV files for the word *go* are located under `/SimulatingCochlearImplants/GSC/go`). For the TIMIT corpus, create a new folder __TIMIT__ and move the corpus files to this folder (so that the training data are located under `/SimulatingCochlearImplants/TIMIT/timit/train`).


## Run the experiments 

The project's root directory contains Python and R scripts, numbered 1 through 6, which you need to run one after the other in order to carry out the experiments.

*1-featurize.py*      
Featurize the data: Convert the WAV files from both data sets into spectrograms (high-resolution, medium-resolution, and low-resolution). The featurized TIMIT corpus will be saved to `/SimulatingCochlearImplants/Featurize/featurized/gender`, and the featurized google speech commands will be saved to `/SimulatingCochlearImplants/Featurize/featurized/words`.

*2-run_models.py*  
Train the neural networks. Trained models are written to `/SimulatingCochlearImplants/results/models`,
and performance metrics for each epoch are written to SQL data bases kept here: `/SimulatingCochlearImplants/results/data_bases`

*3-run_art.py*  
Run statistical tests to compare performance across models via approximate randomization testing (ART).
Test results are written to CSV files here: `/SimulatingCochlearImplants/results/art`. Many thanks to [Stephan Tulkens](https://github.com/stephantul), who developed the ART script we are using.

*4-plot_accuracy.R*  
This R script reads results from the SQL data bases, plots accuracy over epochs for the different networks, and stores each plot here: `/SimulatingCochlearImplants/results/plots`. (When running this and the following R script, make sure that the project folder is set as the working directory).

*5-inspect_art_results.R*  
A simple R script that lets you inspect the output from our statistical tests (kept in CSV files). 
We found it convenient to use R for this purpose, but you could also use another method to inspect the content of CSV files.

*6-plot_spectrograms.py*  
Plot some of the spectrograms (e.g. to visually compare high-, medium-, and low-resolution spectrograms). The plots are written to `/SimulatingCochlearImplants/results/plots/gender_spectrograms` and `/SimulatingCochlearImplants/results/plots/words_spectrograms`.
