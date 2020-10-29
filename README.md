# Biased TextRank
This repository contains code and data for our paper: 
**Biased textrank: Unsupervised Graph-Based Content Extraction: Ashkan Kazemi, Veŕonica Pérez-Rosas, and Rada Mihalcea. COLING 2020**.

# Requirements
To install the required packages for running the codes on your machine, please run ``pip install -r requirements.txt``
first. 

# Content
* ``/data/``: This directory contains the two datasets used in the experiments. The ``/data/liar/`` directory contains files
for the LIAR-PLUS dataset. The ``/data/us-presidential-debates/``  directory contains the novel presidential debates 
dataset described in the paper.
* ``/src/`` This directory contains implementations of the described experiments in the paper. To run the *biased summarization*
experiment, run ``/src/biased_summarization.py``. For the explanation extraction experiment, run ``explanation_generation.py``.  
