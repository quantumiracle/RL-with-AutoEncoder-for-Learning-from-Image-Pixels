# Reinforcement Learning with Convolutional AutoEncoder for Learning from Image Pixels

## Basics:

Algorithm: Soft Actor-Critic

Environment: Reacher

Settings: the observation is encoded screenshot of the env. The screenshot is of size (200, 200, 3) as a downsampled RGB image, and through a convolutional autoencoder it becomes a low-dimensional vetor.

<p align="center">
<img src="https://github.com/quantumiracle/End-To-End-RL-with-AutoEncoder/blob/master/img/data_2.png" width="50%">
</p>

Document about experiments and results is [here](https://github.com/quantumiracle/RL-with-AutoEncoder-for-Learning-from-Image-Pixels/blob/master/end2end_sac_ae_on_reacher(1).pdf).


## To Run:

1. `python ./vae/env_2.py` for generating image samples;

2. `python ./vae/ae.py` for pre-training the AE;

3. `python sac.py` fortraning the SAC.

Remember to make the model path and name correct!
