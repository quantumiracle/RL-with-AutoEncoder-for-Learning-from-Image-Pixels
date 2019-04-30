# AE/VAE for Image Compression

`env_2.py`: generate screenshot images data;

`cnn_vae.py`: Tensorflow version of CNN VAE;

`ae.py`: Pytorch version of AE/VAE, Conv AE/VAE;



Results with Pytorch version code: 

* Convolution based AE/VAE has much better results than dense layers only, Convolution based AE can almost recover the original images (final loss 0.02833); Convolution based VAE cannot learn the variance of different images, different latent codes have almost the same generated images.

\begin{equation}

ddf

\end{equation}