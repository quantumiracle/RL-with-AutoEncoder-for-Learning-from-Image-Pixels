import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.distributions import normal
import argparse
# from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)

args = parser.parse_args()

f =gzip.open('./screenshot_data2002003_2.gzip','rb')
save_file='./model/ae.ckpt'

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

# We set a random seed to ensure that your results are reproducible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


def denorm_for_sigmoid(x):
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), X_channel, X_dim, X_dim)
    return x
denorm = denorm_for_sigmoid

# train_dat = datasets.MNIST(
#     "data/", train=True, download=True, transform=transform
# )
# test_dat = datasets.MNIST("./data/", train=False, transform=transform)

if not os.path.exists('./single_layer_AE'):
    os.mkdir('./single_layer_AE')
    
learning_rate = 2e-3  # 5e-3 for conv_ae
latent_dim = 64 # dim of encoded representation
X_dim = 200 # dim of the input X_dim*X_dim
X_channel = 3 # input channel
num_epochs = 300 # number of training epochs
data_samples=800 # total number of training samples
batch_size = 20 # batch size of training for each epoch
VAE=False # VAE if true, else AE
CONV=True # Convolution if true

# train_loader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=16)
# test_loader = DataLoader(test_dat, batch_size, shuffle=False, num_workers=16)
# it = iter(test_loader)
# sample_inputs, _ = next(it)

fixed_input=[]
num_test=2 # number of test samples
for i in range(num_test):
    fixed_input.append(pickle.load(f)[:,:,0:0+X_channel]/255.)
    jump_sample=500 # jump samples in case use neighboring samples for test
    for jump in range (jump_sample):
        pickle.load(f)
fixed_input=np.transpose(np.array(fixed_input), (0,3,1,2)) # switch from (x-dim,x-dim,x-channel) to (x-channel,x-dim,x-dim)
fixed_input = torch.Tensor(fixed_input).view(num_test,X_channel,X_dim,X_dim)

save_image(fixed_input, './single_layer_AE/image_original.png')


class AE_single(nn.Module):

    def __init__(self, in_dim=X_channel*X_dim*X_dim, h_dim=32):
        super(AE_single, self).__init__()
        h_dim2 = 2*h_dim
        self.CONV_NUM_FEATURE_MAP=8
        self.CONV_KERNEL_SIZE=4
        self.CONV_STRIDE=2
        self.CONV_PADDING=1

        if VAE:
            # Encoder
            if CONV:
                self.encoder=nn.Sequential(
                nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
                nn.LeakyReLU(0.2, inplace=True),
                )
                # output_size = (i-k+2p)//2+1
                h_dim2 = int(self.CONV_NUM_FEATURE_MAP*2*(X_dim/(self.CONV_STRIDE*self.CONV_STRIDE))**2)
            else: # dense layers only
                self.encoder=nn.Sequential(
                    nn.Linear(in_dim, h_dim2),
                    nn.ReLU()
                )
            self.mu=nn.Linear(h_dim2,h_dim)
            self.log_var=nn.Linear(h_dim2,h_dim)

            # Decoder
            if CONV:
                self.decoder1 = nn.Sequential(
                nn.Linear(h_dim, h_dim2),
                nn.LeakyReLU(0.2, inplace=True))
                self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d( self.CONV_NUM_FEATURE_MAP * 2, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.CONV_NUM_FEATURE_MAP, X_channel, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                nn.BatchNorm2d(X_channel),
                nn.Sigmoid()
                )
            else: # dense layers only
                self.decoder = nn.Sequential(
                nn.Linear(h_dim, h_dim2),
                nn.ReLU(),
                nn.Linear(h_dim2, in_dim),
                nn.Sigmoid()
                )
        else: # AE only
            # Encoder
            if CONV:
                # output_size = (i-k+2p)//2+1
                h_dim2 = int(self.CONV_NUM_FEATURE_MAP*2*(X_dim/(self.CONV_STRIDE*self.CONV_STRIDE))**2)
                self.encoder=nn.Sequential(
                nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
                nn.LeakyReLU(0.2, inplace=True)
                )
                self.z=nn.Linear(h_dim2, h_dim)



            else:
                self.encoder = nn.Linear(in_dim, h_dim)
            ''' add layer '''
            # self.encoder = nn.Sequential(
        #     nn.Linear(in_dim, h_dim2),
        #     nn.Linear(h_dim2, h_dim)
        #     )

            # Decoder
            if CONV:
                self.decoder1 = nn.Sequential(
                nn.Linear(h_dim, h_dim2),
                nn.LeakyReLU(0.2, inplace=True))
                self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d( self.CONV_NUM_FEATURE_MAP * 2, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.CONV_NUM_FEATURE_MAP, X_channel, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                nn.BatchNorm2d(X_channel),
                nn.Sigmoid()
                )
            else:
                self.decoder = nn.Sequential(
                nn.Linear(h_dim, in_dim),
                nn.Sigmoid()
                )
            ''' add layer '''
            # self.decoder = nn.Sequential(
        #     nn.Linear(h_dim, h_dim2),
        #     nn.Linear(h_dim2, in_dim),
        #     nn.Sigmoid()
        #     )
        
 
    def encode(self, x):
        if CONV: # switch input form (N, x_channel*x_dim*x_dim) to (N, x_channel, x_dim, x_dim) for convolution
            x=x.view(-1, X_channel, X_dim, X_dim)
        if VAE:
            x=self.encoder(x)
            x=x.view(x.size(0),-1)  # flatten after convolution for the subsequent dense layer
            mu=self.mu(x)
            log_var=self.log_var(x)
            return mu, log_var
        else:
            if CONV:
                x=self.encoder(x)
                x=x.view(x.size(0),-1) # flatten after convolution for the subsequent dense layer
                z=self.z(x)
            else:
                z = self.encoder(x)
            return z
    
    def decode(self, z):
        if CONV:
            z1=self.decoder1(z)
            # view as (N, n_featuremap, dim, dim) after dense layer for subsequent convolution
            z1=z1.view(-1, self.CONV_NUM_FEATURE_MAP*2, int(X_dim/(self.CONV_STRIDE*self.CONV_STRIDE)), int(X_dim/(self.CONV_STRIDE*self.CONV_STRIDE)) )
            x_recon=self.decoder2(z1)
            x_recon=x_recon.view(x_recon.size(0),-1)
        else:
            x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        if VAE:
            mu, log_var=self.encode(x)
            z=self.sample_z(mu,log_var)
            x_recon = self.decode(z)
            kl_loss=self.kl_loss(mu, log_var)
            return x_recon, kl_loss

        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon

    def sample_z(self, mu, log_var):
        dis=normal.Normal(mu, torch.exp(log_var))
        ''' 
        sample() may have no error for running, but does not calculate gradients
        only rsample() with parameterization trick can calculate and backpropagate gradients through samples
        '''
        return dis.rsample()  

    def kl_loss(self, mu, log_var):
        kl_loss=0.5* torch.mean(torch.exp(log_var)+mu**2-1.-log_var)
        return kl_loss


single_layer_AE = AE_single(h_dim=latent_dim)

# criterion = nn.L1Loss(reduction='sum')
criterion = nn.L1Loss()

def loss_function_AE(recon_x, x):
    recon_loss = criterion(recon_x, x)
    return recon_loss


model = single_layer_AE.to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))  # what would the number actually be
head_str=''
if CONV: 
    head_str+='Convolution '
if VAE:
    head_str+='Variational'
print(head_str, model)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# model.train()

if args.train:
    total_batch=[]
    for j in range (int(data_samples/batch_size)):
        for i in range(batch_size):
            sample=pickle.load(f)[:,:,0:0+X_channel]/255.  # image value 0-1, size:(200,200,1)
            sample=np.transpose(sample, (2,0,1)) # switch from (x-dim,x-dim,x-channel) to (x-channel,x-dim,x-dim)
            total_batch.append(sample.reshape(-1))  # (200*200,)

    for epoch in range(num_epochs):
        train_loss = 0
        train_kl_loss=0
        train_recon_loss=0
        for j in range (int(data_samples/batch_size)):
            batch=total_batch[j*batch_size:(j+1)*batch_size]
            batch = torch.Tensor(batch).to(device)
            optimizer.zero_grad()
            # forward
            if VAE:
                recon_batch, kl_loss = model(batch)
                recon_loss=loss_function_AE(recon_batch, batch)
                loss = recon_loss + kl_loss
                
            else:  
                recon_batch = model(batch)
                recon_loss=loss_function_AE(recon_batch, batch)
                kl_loss=0.
                loss = recon_loss
            # backward
            loss.backward()
            train_loss += loss.item()
            train_kl_loss +=  kl_loss
            train_recon_loss += recon_loss
            optimizer.step()
        # print out losses and save reconstructions for every epoch
        # print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_loader.dataset)))
        if VAE:
            z_dis = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            z=z_dis.sample((fixed_input.size(0), latent_dim))
            recon = model.decode(z.view(fixed_input.size(0),-1).to(device))
            # recon,_ = model(fixed_input.view(fixed_input.size(0), -1).to(device))
        else:
            recon = model(fixed_input.view(fixed_input.size(0), -1).to(device))
        
        recon = denorm(recon.cpu())
        save_image(recon, './single_layer_AE/reconstructed_epoch_{}.png'.format(epoch))
        # save the model
        torch.save(model.state_dict(), './single_layer_AE/model.pth')
        print('epoch: {}, loss: {:.4}, kl_loss: {:.4}, recon_loss: {:.4}'.format(epoch, train_loss, train_kl_loss, train_recon_loss))
    f.close()


if args.test:
    model.load_state_dict(torch.load("./single_layer_AE/model.pth"))
    TEST_SAMPLES=10
    MUM_IMG=1
    for i in range(TEST_SAMPLES):
        if VAE:
            z_dis = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            z=z_dis.sample((MUM_IMG, latent_dim))
            recon = model.decode(z.view(MUM_IMG,-1).to(device))
            # recon,_ = model(fixed_input.view(fixed_input.size(0), -1).to(device))
        else:
            recon = model(fixed_input.view(fixed_input.size(0), -1).to(device))
        recon = denorm(recon.cpu())
        save_image(recon, './test/reconstructed_img_{}.png'.format(i))
# # load the model
# model.load_state_dict(torch.load("./single_layer_AE/model.pth"))
# model.eval()
# test_loss = 0
# with torch.no_grad():
#     for i, (img, _) in enumerate(test_loader):
#         img = img.view(img.size(0), -1)
#         img = img.to(device)
#         recon_batch = model(img)
#         test_loss += loss_function_AE(recon_batch, img)
#     # reconstruct and save the last batch
#     recon_batch = model(recon_batch.view(recon_batch.size(0), -1).to(device))
#     img = denorm(img.cpu())
#     # save the original last batch
#     save_image(img, './single_layer_AE/test_original.png')
#     save_image(denorm(recon_batch.cpu()), './single_layer_AE/reconstructed_test.png')
#     # loss calculated over the whole test set
#     test_loss /= len(test_loader.dataset)
#     print('Test set loss: {:.4f}'.format(test_loss))



# # visualize the original images of the last batch of the test set
# img = make_grid(img, nrow=4, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
# show(img)


# # visualize the reconstructed images of the last batch of test set
# recon_batch_denorm = denorm(recon_batch.view(-1, 1, 28, 28).cpu())
# print(recon_batch_denorm.size())
# recon_batch_denorm = make_grid(recon_batch_denorm, nrow=4, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
# show(recon_batch_denorm)