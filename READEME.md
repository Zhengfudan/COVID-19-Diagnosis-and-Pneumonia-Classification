# Introduction

This is the official implement of **COVID-19-Diagnosis-and-Pneumonia-Classification**. 

We only release the code and weights the SENet model of **multi-class classification**, without the dataset we used.(In order to protect the privacy of covid-19 patients.)

Therefore, if you want to use our model to **train** or **evaluate** on your covid-19 dataset. The following description will help you.

# Environment

The code is developed using python 3.6. The code is developed and tested using 4 NVIDIA GPU cards. Other platforms or GPU cards are not fully tested.

# Implementation

1. Install pytorch >= v1.0.0

2. clone this repo

3. ```
   pip install -r requirements.txt
   ```

## Train

- run ``` python train.py --data_path your_2d_CT_images_dataset```, `your_2d_CT_images_dataset` is the path of your training dataset. 
- There're a lot of other parameters like `learning rate` , `epochs`, `batch_size`. (see the file--`train.py`), you can edit it to better train your own model.#

# Evaluate

- run ``` python my_evaluate.py --data_path your_2d_CT_images_dataset --checkpoints_path checkpoints/Best_model.pth ```, `your_2d_CT_images_dataset` is the path of your test dataset. 

# visualization

- run ```visualization/visualization.py`--data_path your_2d_CT_images_dataset --checkpoints_path checkpoints/Best_model.pth ```.`your_2d_CT_images_dataset` is the path of the dataset you want to visualize.

