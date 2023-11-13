# Helicobacter pylori Detection
## Introduction
The aim of this project is to generate an algorithm capable to detect Helicobacter pylori (a bacteria that is located in the stomach) in biopsy images. To do so, our approach involves the creation of an autoencoder.

## Repository Structure

The project is organized into multiple folders:

- **Extracting the windows:** Here, we can find the code used to obtain the patches used to train and test the autoencoder. In addition, there is a file with examples of the images obtained.
- **Autoencoder:** This file contains the code used to obtain the autoencoder (Dataloader creation, Grid Search for Parameter Optimization, the evaluation phase...
- **Evaluation:** In tis folder we can find the red_pixel_calc.py file in which we can find the evaluation, we pass the patches through the trained autoencoder and then calculate the red pixels difference between the input and the output. The information for each patient is stored in three pickles, one for healthy and two for infected ones (ALTA/BAIXA) which will be treated equally in the future stages. In addition, we can find Evaluation_with_roc_curve.ipynb file, where we can find the final predictions of the patients' results, based on the computation of the red pixels' difference from the previous file, and the results of the metrics used to evaluate the performance of the model using a roc_curve function.
- **Presentation:** This file contains the presentation of the project.

## Contributors
Júlia Garcia
Guillem Samper
Nerea Qing Muñoz
