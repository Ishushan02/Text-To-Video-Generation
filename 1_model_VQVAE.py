import torch
import torch.nn as nn
from torchview import draw_graph
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional  as Fn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from io import BytesIO
from IPython.core.display import HTML
from IPython.display import Image as IPyImage, display
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import os
from piq import ssim
import gc
import pandas as pd
import urllib
import io
import cv2
import hashlib

import wandb
wandb.login()

wandb.init(
    project="T2V-VQVAE",  
    name="experiment-1",    
    id="47t9tup8",  
    resume="allow",
    # config={                       
    #     "epochs": 1000,
    #     "batch_size": 64,
    # }
)

class VectorQuantizeImage(nn.Module):
    def __init__(self, codeBookDim = 64, embeddingDim = 32, decay = 0.99, eps = 1e-5):
        super().__init__()

        self.codeBookDim = codeBookDim
        self.embeddingDim = embeddingDim
        self.decay = decay
        self.eps = eps
        self.dead_codeBook_threshold = codeBookDim * 0.6

        self.codebook = nn.Embedding(codeBookDim, embeddingDim)
        nn.init.xavier_uniform_(self.codebook.weight.data)

        self.register_buffer('ema_Count', torch.zeros(codeBookDim))
        self.register_buffer('ema_Weight', self.codebook.weight.data.clone())

    def forward(self, x):
        x_reshaped = x.view(-1, self.embeddingDim)

        distance = (torch.sum(x_reshaped**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(x_reshaped, self.codebook.weight.t()))
        
        encoding_indices = torch.argmin(distance, dim=1) 
        encodings = Fn.one_hot(encoding_indices, self.codeBookDim).type(x_reshaped.dtype)
        quantized = torch.matmul(encodings, self.codebook.weight)

        if self.training:
            self.ema_Count = self.decay * self.ema_Count + (1 - self.decay) * torch.sum(encodings, 0)
            
            x_reshaped_sum = torch.matmul(encodings.t(), x_reshaped.detach())
            self.ema_Weight = self.decay * self.ema_Weight + (1 - self.decay) * x_reshaped_sum
            
            n = torch.clamp(self.ema_Count, min=self.eps)
            updated_embeddings = self.ema_Weight / n.unsqueeze(1)
            self.codebook.weight.data.copy_(updated_embeddings)

        
        avg_probs = torch.mean(encodings, dim=0)
        log_encoding_sum = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(log_encoding_sum)

        entropy = log_encoding_sum
        normalized_entropy = entropy / torch.log(torch.tensor(self.codeBookDim, device=x.device))
        diversity_loss = 1.0 - normalized_entropy

        return quantized, encoding_indices, perplexity, diversity_loss
        
        
# vq = VectorQuantizeImage(codeBookDim=64,embeddingDim=32)
# rand = torch.randn(1024,32)
# vq(rand)

class VecQVAE(nn.Module):
    def __init__(self, inChannels = 1, hiddenDim = 32, codeBookdim = 128, embedDim = 128):
        super().__init__()
        self.inChannels = inChannels
        self.hiddenDim = hiddenDim
        self.codeBookdim = codeBookdim
        self.embedDim = embedDim

        self.encoder = nn.Sequential(
            nn.Conv2d(inChannels, hiddenDim, 4, 2, 1), 
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2 * hiddenDim, 2 * hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2 * hiddenDim, embedDim, 1),
        )

        self.vector_quantize = VectorQuantizeImage(codeBookDim=codeBookdim,embeddingDim=embedDim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(2 * hiddenDim, 2 * hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2 * hiddenDim, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hiddenDim, inChannels, 1),
            nn.Sigmoid()
        )

    def encodeImage(self, x, noise_std = 0.15):
        if self.training:
            encodedOut = self.encoder(x)
            encodedOut = encodedOut + torch.randn_like(encodedOut) * noise_std
        else:
            encodedOut = self.encoder(x)

        return encodedOut

    def decodeImage(self, quantized_vector):
        decodedOut = self.decoder(quantized_vector)
        return decodedOut

    def forward(self, x):
        batch_size, time_frame, inChannels, height, width = x.shape

        x_frames = rearrange(x, 'b t c h w -> (b t) c h w')
        encodedOut = self.encodeImage(x_frames)
        batch_size_time_frame, encoded_channel, encoded_height, encoded_width = encodedOut.shape
        
        # print(f"Encoded Shape: {encodedOut.shape}")

        
        vectorize_input = rearrange(encodedOut, 'bt d h w -> (bt h w) d')
        quantized_vectors, encoding_indices, perplexity, diversity_loss  = self.vector_quantize(vectorize_input)
        codebook_loss = Fn.mse_loss(vectorize_input.detach(), quantized_vectors)
        commitment_loss = Fn.mse_loss(vectorize_input, quantized_vectors.detach())

        quantized_vectors = vectorize_input + (quantized_vectors - vectorize_input).detach()
        # print(f"CodeBook Loss: {codebook_loss} , Commitment Loss: {commitment_loss}")
        # print(f"Quantized SHape: {quantized_vectors.shape}")

        decoder_input = rearrange(quantized_vectors, '(bt h w) d -> bt d h w', bt = batch_size_time_frame, d = encoded_channel, h = encoded_height, w = encoded_width)
        # print(f"Decoded Input SHape: {decoder_input.shape}")
        decodedOut = self.decodeImage(decoder_input)

        # print(f"Decoded SHape: {decodedOut.shape}")
        
        return decoder_input, decodedOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss

# VQ = VecQVAE(inChannels = 3, hiddenDim = 256, codeBookdim = 128, embedDim = 64)
# test = torch.randn(32, 10, 3, 64, 64)
# quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = VQ(test)
# quantized_latents.shape, decoderOut.shape, codebook_loss, commitment_loss, encoding_indices.shape, perplexity, diversity_loss


dataset = pd.read_csv("./data/modified_tgif.csv")
dataset = dataset[(dataset['frames'] <= 40) & (dataset['frames'] > 15)].copy().reset_index(drop=True)
dataset = dataset[:10000] # thread 1 first 10000
# dataset.shape

def getNumpyArray(dataset, index):
    url = dataset['url'][index]
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR_RGB)

    return image

# tImg = getNumpyArray(dataset, 10)
# tImg.shape

def getNumpyArray(dataset, index):
    url = dataset.iloc[index]['url']
    resp = urllib.request.urlopen(url)
    data = resp.read()
    pil_image = Image.open(io.BytesIO(data))
    
    frames = []
    for frame in ImageSequence.Iterator(pil_image):
        frame = frame.convert("RGB")
        frame_np = np.array(frame)
        frames.append(frame_np)
    
    frames = np.array(frames)
    return frames


# tImg = getNumpyArray(dataset, 74801)
# tImg.shape


class FrameDataset(Dataset):
    def __init__(self, data, totalSequence = 40, transform = None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.totalSequence = totalSequence

    def __len__(self):
        return len(self.data)
    
    def npArray(self, index):
        try:
            row = self.data.iloc[index]
            totalframes = self.data.iloc[index]['frames']
            url = row['url']
            resp = urllib.request.urlopen(url)
            image_data = resp.read()
            img = Image.open(io.BytesIO(image_data))
    
            frames = []
            for frame in ImageSequence.Iterator(img):
                frame_rgb = frame.convert("RGB")
                frames.append(np.array(frame_rgb))
    
            return frames
    
        except Exception as e:
            print(f"Error processing index {index} for url {url}: {e}")
            fallback = torch.zeros((256, 256, 3), dtype=torch.uint8)
            return [fallback.numpy()]
    
    def __getitem__(self, index):
        # print(index)
        gif = self.npArray(index)
        caption = self.data.iloc[index]['caption']
        totalframes = len(gif)#self.data.iloc[index]['frames']
        
        if totalframes < self.totalSequence:
            gif += [gif[-1]] * (self.totalSequence - totalframes)

        tensorFrames = torch.stack([
            self.transform(Image.fromarray(frame)) for frame in gif
        ])

        tensorFrames = tensorFrames/255.0

        return tensorFrames, caption
    

tranform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# fdata = FrameDataset(dataset, transform=tranform)

# X, Y = fdata.__getitem__(74801)
# print(X.shape, Y)

BATCH_SIZE = 2
codeBookdim = 256
embedDim = 128
hiddenDim = 256
inChannels = 3
tranform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
torchDataset = FrameDataset(dataset, transform=tranform)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = DataLoader(torchDataset, batch_size=BATCH_SIZE, shuffle = True)
modelA = VecQVAE(inChannels = inChannels, hiddenDim = hiddenDim, codeBookdim = codeBookdim, embedDim = embedDim).to(device)
lossFn = nn.MSELoss()
optimizerA = torch.optim.Adam([
                    {'params': modelA.encoder.parameters(), 'lr': 2e-4},
                    {'params': modelA.decoder.parameters(), 'lr': 2e-4},
                    {'params': modelA.vector_quantize.parameters(), 'lr': 1e-4}
                ], weight_decay=1e-5)
schedulerA = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizerA, T_0=10, T_mult=2, eta_min=1e-6
            )

epochs = 1000


# modelValA = torch.load("./projects/t2v-gif/models/VQVAE-GIF.pt", map_location=torch.device('cpu'))
# modelA.load_state_dict(modelValA)

for each_epoch in range(epochs):
    modelA.train()
    reconstruct_loss = 0.0
    codeb_loss = 0.0
    commit_loss = 0.0
    vqvaeloss = 0.0
    diverse_loss = 0.0
    ssim_loss = 0.0
    
    loop = tqdm(dataloader, f"{each_epoch}/{epochs}")
    perplexities = []

    for X, caption in loop:
        X = X.to(device)
        # Y = Y.to(device)
        
        quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = modelA(X)

        # print(X.shape, decoderOut.shape)
        X = rearrange(X, 'b t d h w -> (b t) d h w', b = BATCH_SIZE, t = 40, d = 3, h = 128, w = 128)
        
        ssim_score = ssim(X, decoderOut, data_range=1.0)
        ssim_loss = 1.0 - ssim_score

        reconstruction_loss = torch.mean((X - decoderOut)**2)
        
        loss = reconstruction_loss + codebook_loss + 0.2 * commitment_loss + 0.1 * diversity_loss + 0.1 * ssim_loss
        vqvaeloss += loss.item()

        
        reconstruct_loss += reconstruction_loss.item()
        diverse_loss += diversity_loss.item()
        codeb_loss += codebook_loss.item()
        commit_loss += commitment_loss.item()
        ssim_loss += ssim_loss.item()
        perplexities.append(perplexity)
        
        
        optimizerA.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelA.parameters(), max_norm=1.0)
        optimizerA.step()
        loop.set_postfix({"TotalL": f"{vqvaeloss}", "ReconsL": f"{reconstruct_loss}", "CodeL":f"{codeb_loss}",
                          "CommitL":f"{commitment_loss}", "Perplexity":f"{perplexity}", "Diversity Loss":f"{diverse_loss}", "SSIM Loss":f"{ssim_loss}"})
    #     break
    # break

    average_perplexity = sum(perplexities)/len(perplexities)
    vqvaeloss /= len(dataloader)   
    reconstruct_loss /= len(dataloader)   
    codeb_loss /= len(dataloader)   
    commit_loss /= len(dataloader)   
    diverse_loss /= len(dataloader)
    torch.save(modelA.state_dict(), "./models/VQVAE-GIF.pt")
    wandb.log({
        "Epoch": each_epoch,
        "VQVAE LR": optimizerA.param_groups[0]['lr'],
        "VQVAE Loss": vqvaeloss,
        "Reconstruction Loss": reconstruct_loss,
        "Codebook Loss": codeb_loss,
        "Commitment Loss": commit_loss,
        "Diversity Loss": diverse_loss,
        "Perplexity": average_perplexity,
        "SSIM Loss":ssim_loss,
    })
    schedulerA.step()
 