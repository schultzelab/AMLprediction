# Scalable diagnosis of acute myeloid leukemia using high-dimensional machine learning and blood transcriptomics

This repository contains the code necessary to reproduce the main figures from the manuscript by Warnat-Herresthal et. al 2019. 

You can use our docker-container https://hub.docker.com/r/schultzelab/aml_classifier, which provides the data and all packages necessary to reproduce the calculations. For reproducing the figures, we provide our main results in the folder `results`. 

The folders contain the following scripts: 

- calculations: R-scripts to reproduce the main prediction results. 
- figures: RMD-files and knitted HTMLs with figures and subfigures. 
- results: output directory for the calculation scripts and our precalculated results to reproduce the figures.
- docker: The dockerfile for https://hub.docker.com/r/schultzelab/aml_classifier

The scripts were written by Stefanie Warnat-Herresthal, Konstaninos Perrakis and Bernd Taschler. 
