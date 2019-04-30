# RL with Screenshot as Inputs

DDPG with direct screenshot as inputs. 

Raw code: the observation is screenshot of size (200, 200, 3) as a downsampled RGB image, and both the actor and critic have a cnn+fc nework. It cannot work!

To Run:

`python -m run --alg=ddpg --num_timesteps=1e4 --train/test`

