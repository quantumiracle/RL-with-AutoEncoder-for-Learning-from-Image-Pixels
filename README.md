# Reinforcement Learning with Convolutional AutoEncoder for End-to-End Learning from Images

Algorithm: Soft Actor-Critic

Environment: Reacher

Settings: the observation is encoded screenshot of the env. The screenshot is of size (200, 200, 3) as a downsampled RGB image, and through a convolutional autoencoder it becomes a low-dimensional vetor.

<p align="center">
<img src="https://github.com/quantumiracle/End-To-End-RL-with-AutoEncoder/blob/master/img/data_2.png" width="50%">
</p>

To Run:

1. `python ./vae/env_2.py` for generating image samples;

2. `python ./vae/ae.py` for pre-training the AE;

3. `python sac.py` fortraning the SAC.

Remember to make the model path and name correct!
