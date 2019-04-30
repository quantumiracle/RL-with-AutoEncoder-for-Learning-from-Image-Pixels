from __future__ import division
import argparse
import matplotlib.pyplot as plt
import pickle
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.gridspec as gridspec
import os
# from tensorflow.examples.tutorials.mnist import input_data
# np.set_printoptions(threshold=np.inf)

f =gzip.open('./screenshot_data2002003.gzip','rb')
save_file='./model/vae.ckpt'

z_dim = 500
X_dim = 200
X_channel = 1
conv_dim = 32
h_dim = 128

VAE=False # VAE if true, else AE
CONV=True # convolution if true, else dense layers only
#lr = 1e-4

def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None,X_dim,X_dim,X_channel])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
lr = tf.placeholder(tf.float32)

if CONV:
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):

        Q_W1 = tf.Variable(xavier_init([int(X_dim*X_dim/((2*2))*conv_dim), h_dim]))
        Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
        Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

        Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
        Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))




    def Q(X):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # X = tf.reshape(X, [-1, X_dim, X_dim, 3])
            conv = tf.contrib.layers.conv2d(X,
                                            conv_dim,
                                            [5, 5],
                                            (2, 2),
                                            padding='SAME',
                                            activation_fn=lrelu,
                                            normalizer_fn=tf.contrib.layers.batch_norm)
            conv = tf.contrib.layers.conv2d(conv,
                                            conv_dim,
                                            [5, 5],
                                            (1, 1),
                                            padding='SAME',
                                            activation_fn=lrelu,
                                            normalizer_fn=tf.contrib.layers.batch_norm)
            flat = tf.contrib.layers.flatten(conv)
            #print(flat.shape)
            h = tf.nn.relu(tf.matmul(flat, Q_W1) + Q_b1)
            z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
            z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
        return z_mu, z_logvar

else: # dense layers only
    def Q(X):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            X=tf.layers.flatten(X)
            X=tf.layers.dense(X, h_dim, activation=lrelu)
            z_mu=tf.layers.dense(X, z_dim, activation=None)
            z_logvar=tf.layers.dense(X, z_dim, activation=None)
        return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.math.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

if CONV:
    P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    P_W2 = tf.Variable(xavier_init([h_dim, int(X_dim*X_dim/((2*2))*conv_dim)]))
    P_b2 = tf.Variable(tf.zeros(shape=[int(X_dim*X_dim/((2*2))*conv_dim)]))


    def P(z):
        h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
        logits = tf.matmul(h, P_W2) + P_b2
        logits=tf.reshape(logits, [-1,int(X_dim/2),int(X_dim/2),conv_dim])
        trans_conv = tf.contrib.layers.conv2d_transpose(logits,
                                                        conv_dim, 
                                                        [5, 5],
                                                        (1, 1),
                                                        padding='SAME',
                                                        activation_fn=lrelu,
                                                        normalizer_fn=tf.contrib.layers.batch_norm)

        trans_conv = tf.contrib.layers.conv2d_transpose(trans_conv,
                                                        X_channel,  # output dim, 3 for 3-channel image
                                                        [5, 5],
                                                        (2, 2),
                                                        padding='SAME',
                                                        # activation_fn=lrelu,
                                                        activation_fn=tf.nn.sigmoid,
                                                        normalizer_fn=tf.contrib.layers.batch_norm)

        # out = tf.nn.sigmoid(trans_conv)
        # out = tf.nn.relu6(trans_conv)/6.
        # out =  tf.nn.relu(trans_conv)
        out = trans_conv
        
        return out, logits

else: # dense layers only
    def P(z):
        z=tf.layers.dense(z, h_dim, activation=lrelu)
        logits=tf.layers.dense(z, X_dim*X_dim*conv_dim, activation=lrelu)
        out=tf.nn.sigmoid(logits)
        out=tf.reshape(out, [-1, X_dim, X_dim, X_channel])

        return out, logits



# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
if VAE:
    out, logits = P(z_sample)
else:
    out, logits = P(z_mu)


# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
# recon_loss = tf.reduce_sum(tf.abs(out -  X))
recon_loss=tf.reduce_sum(tf.losses.mean_squared_error(out, X))
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.math.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
#recon_loss=tf.reduce_sum(tf.abs(X -  X))
if VAE:
    # VAE loss
    vae_loss = tf.reduce_mean(recon_loss + kl_loss)
else:
    # AE loss
    vae_loss = tf.reduce_mean(recon_loss)

solver = tf.train.AdamOptimizer(lr).minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if not os.path.exists('convae/'):
    os.makedirs('convae/')


Loss=[]
It=[]
train_times=1000
batch=[]
data_samples=1
epoch_samples=1

# load data
for i in range(data_samples):
    print(i)
    if X_channel>1:  # channel==3
        batch.append(pickle.load(f)/255.) # rgb image value range 0-255
    else: # channel==1
        batch.append(pickle.load(f)[:,:,1:2]/255.) # rgb image value range 0-255

print(np.array(batch).shape)

# save original img
if X_channel>1:
    plt.imshow(batch[0])
else:
    plt.imshow(batch[0][:,:,0])
plt.savefig('convae/{}.png'.format(str('origin').zfill(3)), bbox_inches='tight')

# vae training
for it in range(train_times):
    for epo in range(data_samples//epoch_samples):

        _, loss ,recon_l, kl_l, output = sess.run([solver, vae_loss,recon_loss,kl_loss,out], \
        feed_dict={X: batch[epo*epoch_samples:epoch_samples*(epo+1)],lr:1e-3/train_times})
        Loss.append(loss)
        It.append(it)
    
    print('Iter: {}'.format(it))
    #print('Loss: {:.4}'. format(loss),recon_l,kl_l)
    print('Loss: {:.4}, KL: {}, Recon: {}'.format(loss, kl_l, recon_l))


    sample = sess.run(X_samples, feed_dict={z: np.random.randn(1,z_dim)})


    if X_channel>1:
        plt.imshow(sample.reshape(X_dim,X_dim,X_channel))
    else:
        plt.imshow(sample.reshape(X_dim,X_dim))

    plt.savefig('convae/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')


saver.save(sess, save_file)

f.close()