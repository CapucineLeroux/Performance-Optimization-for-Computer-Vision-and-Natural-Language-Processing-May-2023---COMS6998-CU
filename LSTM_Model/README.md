# Optimizing LSTM model with PyTorch on IMBD dataset

Code adapted from https://www.kaggle.com/code/affand20/imdb-with-pytorch

## Content
The original script used in the kaggle notebook is in the file 'Initial Model.ipynb'. It is profiled using the file 'Initial Profiling.ipynb'. The optimizations are made and profiled using the two other files 'Optimized Profiling1.ipynb' for the batch_size and num_workers optimizations, and 'Optimized Profiling2.ipynb' for the pruning optimization. The original dataset and its pre-processed version are in the Data folder. All profiling results, as well as text outputs of scripts and figures and in the Profiling_results folder. The profiling results have been compressed in a zip file to be added to the repo.

## How to run
The notebooks have been run on Google Colab using a V100 GPU and High-RAM config. Remove the google colab drive cells to run it on another computer. 

## Results
The optimal batch size is 128 and the optimal number of workers is 0.
The structured pruning (90 percent) reduces learnable parameters from 34,731,77 to 3,528,662. 
These optimizations induce an overall speed up of x2.


