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
from torchvision.models import vgg16
import kornia
import gc
import pandas as pd
import urllib
import io
import cv2
import hashlib
import wandb

os.environ['K8S_TIMEOUT_SECONDS'] = '12'

wandb.login()

wandb.init(
    project="T2V-VQVAE-2",  
    name="experiment-1-thread-cummlative",    
    id="m6ms1f4w",  
    resume="allow",
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

class SelfAttentionBlock(nn.Module):
    def __init__(self, inChannels, heads = 8):
        super().__init__()
        self.query = nn.Conv2d(inChannels, inChannels // heads, 1)
        self.key = nn.Conv2d(inChannels, inChannels // heads, 1)
        self.value = nn.Conv2d(inChannels, inChannels, 1)

        self.coeff = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channel, height, width = x.shape
        q = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, height * width)
        v = self.value(x).view(batch, -1, height * width)
        attn = torch.matmul(q, k)                                          
        attn = Fn.softmax(attn, dim=-1)
        attn_reshaped = attn.permute(0, 2, 1) 
        
        out = torch.matmul(v, attn_reshaped)                        
        out = out.view(batch, channel, height, width)                    
        out = self.coeff * out + x                                      
        return out

# rad = torch.randn(10, 128, 64, 64)
# sAtt = SelfAttentionBlock(128, 4)
# out = sAtt(rad)
# out.shape

class VecQVAE(nn.Module):
    def __init__(self, inChannels=3, hiddenDim=256, codeBookdim=256, embedDim=128):
        super().__init__()
        self.inChannels = inChannels
        self.hiddenDim = hiddenDim
        self.codeBookdim = codeBookdim
        self.embedDim = embedDim

        self.block1 = nn.Sequential(
            nn.Conv2d(inChannels, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hiddenDim, hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hiddenDim, 2 * hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            SelfAttentionBlock(2 * hiddenDim)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(2 * hiddenDim, 2 * hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True),
            SelfAttentionBlock(2 * hiddenDim)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(2 * hiddenDim, embedDim, 1)
        )

        self.vector_quantize = VectorQuantizeImage(codeBookDim=codeBookdim, embeddingDim=embedDim)

        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(embedDim, 2 * hiddenDim, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(2 * hiddenDim, 2 * hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(2 * hiddenDim),
            nn.ReLU(inplace=True)
        )
        self.block8 = nn.Sequential(
            nn.ConvTranspose2d(2 * hiddenDim, hiddenDim, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True)
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(hiddenDim, hiddenDim, 3, 1, 1),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU(inplace=True)
        )
        self.block10 = nn.Sequential(
            nn.ConvTranspose2d(hiddenDim, hiddenDim // 2, 4, 2, 1),
            nn.BatchNorm2d(hiddenDim // 2),
            nn.ReLU(inplace=True)
        )
       
        self.outputlayer = nn.Sequential(
            nn.Conv2d(hiddenDim // 2, inChannels, 1),
            nn.Sigmoid()
        )

    def encodeImage(self, x, noise_std=0.15):
        if self.training:
            x1 = self.block1(x)
            x2 = self.block2(x1)
            x3 = self.block3(x2)
            x4 = self.block4(x3)
            encoded = self.block5(x4)
            encoded += torch.randn_like(encoded) * noise_std
        else:
            x1 = self.block1(x)
            x2 = self.block2(x1)
            x3 = self.block3(x2)
            x4 = self.block4(x3)
            encoded = self.block5(x4)
        return encoded, (x2, x3, x4)

    def decodeImage(self, quantized_vector, skips):
        x2, x3, x4 = skips
        # print(x2.shape, x3.shape, x4.shape)
        x = self.block6(quantized_vector)
        x = self.block7(x + x4)
        x = self.block8(x + x3)
        x = self.block9(x + x2)
        x = self.block10(x)
        return self.outputlayer(x)

    def forward(self, x):
        batch, timeFrames, channel, height, width = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        encoded, skips = self.encodeImage(x)
        batchTime, encodedChannels, encodedHeight, encodedWidth = encoded.shape
        encoder_reshaped = rearrange(encoded, 'bt d h w -> (bt h w) d')
        quantized_vectors, encoding_indices, perplexity, diversity_loss = self.vector_quantize(encoder_reshaped)

        codebook_loss = Fn.mse_loss(encoder_reshaped.detach(), quantized_vectors)
        commitment_loss = Fn.mse_loss(encoder_reshaped, quantized_vectors.detach())

        quantized_vectors = encoder_reshaped + (quantized_vectors - encoder_reshaped).detach()
        # print(quantized.shape)
        decoder_input = rearrange(quantized_vectors, '(bt h w) d -> bt d h w', bt=batch * timeFrames, d=encodedChannels, h=encodedHeight, w=encodedWidth)
        # print(decoder_input.shape)

        decodedOut= self.decodeImage(decoder_input, skips)
        return decoder_input, decodedOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss


# VQ = VecQVAE(inChannels = 3, hiddenDim = 256, codeBookdim = 128, embedDim = 64)
# test = torch.randn(32, 10, 3, 64, 64)
# quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = VQ(test)
# quantized_latents.shape, decoderOut.shape, codebook_loss, commitment_loss, encoding_indices.shape, perplexity, diversity_loss


dataset = pd.read_csv("./projects/t2v-gif/data/modified_tgif.csv")
dataset = dataset[(dataset['frames'] <= 40) & (dataset['frames'] > 15)].copy().reset_index(drop=True)
print(dataset.shape)
# dataset = dataset[72000:] # 9th thread 
# dataset.shape

'''
The Below function is just for last training run for VQVAE - Cummulative Run I am doing now, remove this for general Runs
'''
threads = []
i = 0
while i < dataset.shape[0]:
    threads.append(dataset[i:i+10000])
    i = i + 10000

cummulativeIndices = []

for i in range(len(threads)):
    indices = threads[i].index[threads[i]['frames'] > 35].tolist()
    cummulativeIndices.extend(indices)

# print(len(cummulativeIndices))
cummulativeData = dataset.loc[cummulativeIndices]
cummulativeData = cummulativeData.reset_index(drop=True)

print(f"Cummulative Dataset Shape: {cummulativeData.shape}")

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

CACHEDIR = './projects/t2v-gif/data/cachedData'

class FrameDataset(Dataset):
    def __init__(self, data, totalSequence=40, transform=None, cache_dir=CACHEDIR):
        super().__init__()
        self.data = data
        self.transform = transform
        self.totalSequence = totalSequence
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def npArray(self, index):
        row = self.data.iloc[index]
        url = row['url']
        resp = urllib.request.urlopen(url)
        image_data = resp.read()
        img = Image.open(io.BytesIO(image_data))

        frames = []
        for frame in ImageSequence.Iterator(img):
            frame_rgb = frame.convert("RGB")
            frames.append(np.array(frame_rgb))
        return frames

    def __getitem__(self, index):
        row = self.data.iloc[index]
        url = row['url']
        caption = row['caption']
        totalframes = row['frames']
        gif_path = os.path.join(self.cache_dir, f'{index}.gif')

        if not os.path.exists(gif_path):
            resp = urllib.request.urlopen(url)
            image_data = resp.read()
            with open(gif_path, 'wb') as f:
                f.write(image_data)
        else:
            with open(gif_path, 'rb') as f:
                image_data = f.read()

        img = Image.open(io.BytesIO(image_data))

        frames = []
        for frame in ImageSequence.Iterator(img):
            frame_rgb = frame.convert("RGB")
            frames.append(np.array(frame_rgb))

        if len(frames) < self.totalSequence:
            frames += [frames[-1]] * (self.totalSequence - len(frames))
        else:
            frames = frames[:self.totalSequence]

        if self.transform:
            tensorFrames = torch.stack([
                self.transform(Image.fromarray(frame)) for frame in frames
            ])
            tensorFrames = tensorFrames / 255.0
            return tensorFrames, caption
        else:
            return frames, caption

    
tranform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# fdata = FrameDataset(cummulativeData, transform=tranform)

# save all in prior
# for i in range(len(cummulativeIndices)):
#     index = cummulativeIndices[i]
#     gif_path = os.path.join(CACHEDIR, f'{index}.gif')

#     if not os.path.exists(gif_path):
#         X, Y = fdata.__getitem__(index)
    # print(X.shape, Y)

print("All Data Pre Saved in Local Directory: ", len(os.listdir(CACHEDIR)))

# fdata = FrameDataset(dataset, transform=tranform)

# X, Y = fdata.__getitem__(74801)
# print(X.shape, Y)

BATCH_SIZE = 2
codeBookdim = 1024
embedDim = 256
hiddenDim = 512
inChannels = 3
tranform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
torchDataset = FrameDataset(cummulativeData, transform=tranform)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = DataLoader(torchDataset, batch_size=BATCH_SIZE, shuffle = True)
modelA = VecQVAE(inChannels = inChannels, hiddenDim = hiddenDim, codeBookdim = codeBookdim, embedDim = embedDim).to(device)
lossFn = nn.MSELoss()
optimizerA = torch.optim.Adam(
                [
                    {'params': modelA.parameters(), 'lr': 2e-4},
                    # {'params': modelA.decodeImage.parameters(), 'lr': 2e-4},
                    # {'params': modelA.vector_quantize.parameters(), 'lr': 1e-4}
                ], weight_decay=1e-5)
schedulerA = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizerA, T_0=10, T_mult=2, eta_min=1e-6
            )


epochs = 1000

def perceptualLoss(pred, target):
    vgg = vgg16(pretrained = True).features[:17].eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # print(pred.shape)
    batch, channels, height, width = pred.shape

    pred = pred.view(batch, channels, height, width)
    target = target.view(batch, channels, height, width)

    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)


    vgg_pred = vgg(pred).to(device)
    vgg_true = vgg(target).to(device)

    perceptualoss = Fn.mse_loss(vgg_pred, vgg_true)
    return perceptualoss

# pred = torch.randn(1, 3, 10, 10)
# pred1 = torch.randn(1, 3, 10, 10)
# out = perceptualLoss(pred, pred1)
# out.item()

def lab_color_loss(pred, target):
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)
    loss = Fn.mse_loss(pred_lab, target_lab)
    return loss

# pred = torch.randn(1, 3, 10, 10)
# pred1 = torch.randn(1, 3, 10, 10)
# out = lab_color_loss(pred, pred1)
# out.item()

# modelValA = torch.load("./projects/t2v-gif/models/VQVAE-GIF.pt", map_location=torch.device('cpu'))
# modelA.load_state_dict(modelValA)
start_epoch = 0

checkpoint_path = "./projects/t2v-gif/models/VQVAE-GIF.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    modelA.load_state_dict(checkpoint['model_state_dict'])
    optimizerA.load_state_dict(checkpoint['optimizer_state_dict'])
    schedulerA.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
else:
    print("Loading pretrained model...")
    modelValA = torch.load("./projects/t2v-gif/models/VQVAE-GIF.pt", map_location=torch.device('cpu'))
    modelA.load_state_dict(modelValA)

modelA = torch.nn.DataParallel(modelA)
modelA.to(device)

for each_epoch in range(start_epoch, epochs):
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
        
        quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = modelA(X)
        
        # print(X.shape, decoderOut.shape)
        X = rearrange(X, 'b t d h w -> (b t) d h w', b = BATCH_SIZE, t = 40, d = 3, h = 128, w = 128)
        
        ssim_score = ssim(X, decoderOut, data_range=1.0)
        ssim_loss = 1.0 - ssim_score

        # reconstruction_loss = torch.mean((X - decoderOut)**2)
        reconstruction_loss = Fn.l1_loss(decoderOut, X)
        colorLoss = lab_color_loss(decoderOut, X)
        perceptualoss = perceptualLoss(decoderOut, X)
        
        loss = reconstruction_loss + codebook_loss + 0.2 * commitment_loss + 0.1 * diversity_loss + 0.1 * ssim_loss + 0.1 * perceptualoss + 0.1 * colorLoss
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
        loop.set_postfix({
            "TotalL": f"{vqvaeloss}", 
            "ReconsL": f"{reconstruct_loss}", 
            "CodeL":f"{codeb_loss}",
            "CommitL":f"{commitment_loss}", 
            "Perplexity":f"{perplexity}", 
            "Diversity Loss":f"{diverse_loss}", 
            "SSIM Loss":f"{ssim_loss}",
            "Perceptual Loss":f"{perceptualoss}",
            "Color Loss":f"{colorLoss}"
        })
    #     break
    # break

    average_perplexity = sum(perplexities)/len(perplexities)
    vqvaeloss /= len(dataloader)   
    reconstruct_loss /= len(dataloader)   
    codeb_loss /= len(dataloader)   
    commit_loss /= len(dataloader)   
    diverse_loss /= len(dataloader)
    perceptualoss /= len(dataloader)
    colorLoss /= len(dataloader)
    # torch.save(modelA.state_dict(), "./projects/t2v-gif/models/VQVAE-GIF.pt")
    torch.save({
        'epoch': each_epoch,
        'model_state_dict': modelA.module.state_dict(),
        'optimizer_state_dict': optimizerA.state_dict(),
        'scheduler_state_dict': schedulerA.state_dict()
    }, checkpoint_path)
    
    wandb.log({
        "VQVAE LR": optimizerA.param_groups[0]['lr'],
        "VQVAE Loss": vqvaeloss,
        "Reconstruction Loss": reconstruct_loss,
        "Codebook Loss": codeb_loss,
        "Commitment Loss": commit_loss,
        "Diversity Loss": diverse_loss,
        "Perplexity": average_perplexity,
        "SSIM Loss":ssim_loss,
        "Perceptual Loss":perceptualoss,
        "Color Loss":colorLoss
    })
    schedulerA.step()
 