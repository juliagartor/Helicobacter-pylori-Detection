# Helicobacter pylori Detection
## Introduction
The purpose of this project is to craft an algorithm with the capability to detect Helicobacter pylori, a bacterium situated in the stomach, within biopsy images. Our strategy revolves around constructing an autoencoder proficient in reproducing biopsy images devoid of the bacterium. Subsequently, we utilize its output to identify images containing the bacterium, employing an explained classification method. We delve into diverse architectures, hyperparameters, and classification methods to attain optimal performance. It's important to note that, due to technical and time constraints, the evaluation part remains incomplete, offering room for further improvement.

## Repository Structure
The project is organized into multiple folders:

- **Extracting the windows:** Here, we can find the code used to obtain the patches used to train and test the autoencoder. In addition, there is a folder with some examples of the images obtained.
- **Autoencoder:** This file contains the code used to obtain the autoencoder (Dataloader creation, Grid Search for Parameter Optimization, the evaluation phase...
- **Evaluation:** In tis folder we can find the red_pixel_calc.py file in which we can find the evaluation, we pass the patches through the trained autoencoder and then calculate the red pixels difference between the input and the output. The information for each patient is stored in three pickles, one for healthy and two for infected ones (ALTA/BAIXA) which will be treated equally in the future stages. In addition, we can find Evaluation_with_roc_curve.ipynb file, where we can find the final predictions of the patients' results, based on the computation of the red pixels' difference from the previous file, and the results of the metrics used to evaluate the performance of the model using a roc_curve function.
- **Presentation:** This file contains the presentation of the project.

## Contributors
- Júlia Garcia 
- Guillem Samper
- Nerea Qing Muñoz
