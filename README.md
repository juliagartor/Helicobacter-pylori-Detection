# Helicobacter pylori Detection
## Introduction
The aim of this project is to generate an algorithm capalble to detect Helicobacter pyori (a bacteria that is located in the stomach) in biopsy images. To do so, our approach involves the creation of 

## Repository Structure

The project is organized into multiple folders:

- **Extracting the windows:** Here, we can find the code used to obtain the patches used to train and test the autoencoder. In addition, there is a file with examples of the images obtained.
- **Autoencoder:** This file contains the code used to obtain the autoencoder (Dataloader creation, Grid Search for Parameter Optimization, the evaluation phase...
- **Evaluation:** In tis folder we can find the red_pixel_calc.py file in which we can find the evaluation, we pass the patches through the trained autoencoder and then calculate the red pixels difference between the input and the output. The info for each patient is stored in three pickles, one for healthy and then two for infected ones (ALTA/BAIXA) which will be treated equally in the future stages.
- **Presentation:** This file contains the presentation of the project.
