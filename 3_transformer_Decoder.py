from transformers import CLIPTokenizer, CLIPTextModel
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
from einops import rearrange
import urllib
import io
import torch.nn.functional  as Fn
from PIL import Image, ImageSequence
import pandas as pd
import torch.nn as nn
import os
from transformers import BertTokenizer, BertModel
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import wandb


wandb.login()

wandb.init(
    project="T2V-Decoder",  
    name="experiment-1-thread-1",    
    id="ymjbna0y",  
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


modelA = VecQVAE(inChannels = 3, hiddenDim = 512, codeBookdim = 1024, embedDim = 256)
modelValA = torch.load("./projects/t2v-gif/models/VQVAE-GIF.pt", map_location=torch.device('cpu'))
modelA.load_state_dict(modelValA['model_state_dict'])# test = torch.randn(32, 10, 3, 64, 64)
# quantized_latents, decoderOut, codebook_loss, commitment_loss, encoding_indices, perplexity, diversity_loss = VQ(test)
# quantized_latents.shape, decoderOut.shape, codebook_loss, commitment_loss, encoding_indices.shape, perplexity, diversity_loss

dataset = pd.read_csv("./projects/t2v-gif/data/modified_tgif.csv")
dataset = dataset[(dataset['frames'] <= 40) & (dataset['frames'] > 15)].copy().reset_index(drop=True)
dataset = dataset[:10000] 

# dataset.shape


CACHEDIR = "./projects/t2v-gif/data/cachedData"

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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# fdata = FrameDataset(dataset, transform=tranform)

# X, Y = fdata.__getitem__(1000)
# print(X.shape, Y)


codeBookdim = 1024
embedDim = 256
hiddenDim = 512
inChannels = 3
tranform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# modelA = VecQVAE(inChannels = inChannels, hiddenDim = hiddenDim, codeBookdim = codeBookdim, embedDim = embedDim).to(device)

modelValA = torch.load("./projects/t2v-gif/models/VQVAE-GIF.pt", map_location=torch.device('cpu'))
epochs = 1000


class Text2Video(nn.Module):
    def __init__(self, embedDimension, sequenceLength, codeBookDim, hiddenLayers, heads, feedForwardDim, text_max_length=128, drop=0.15):
        super().__init__()
        self.max_length = text_max_length
        self.embedDimension = embedDimension
        self.codeBookDim = codeBookDim
        self.sequenceLength = sequenceLength

        self.berTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bertModel = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bertModel.parameters():
            param.requires_grad = False
            
        self.hiddenSize = self.bertModel.config.hidden_size
        
        self.textProjection = nn.Linear(self.hiddenSize, embedDimension)
        self.positionalEmbedding = nn.Embedding(self.max_length, embedDimension)
        self.temporalPositionalEmbedding = nn.Embedding(self.sequenceLength, embedDimension)

        self.textMultiAttention = nn.MultiheadAttention(embedDimension, heads, dropout=drop, batch_first=True)
        self.textlayerNorm = nn.LayerNorm(embedDimension)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedDimension, 
            nhead=heads, 
            dim_feedforward=feedForwardDim, 
            dropout=drop, 
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=hiddenLayers)
        self.decoder_norm = nn.LayerNorm(embedDimension)
        self.predictIndices = nn.Linear(embedDimension, codeBookDim)

    def forward(self, text, device):
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, (list, tuple)):
            pass
        else:
            raise ValueError(f"Give string or list of strings, recieved this {type(text)}")
            
        batchSize = len(text)

        tokens = self.berTokenizer(text, return_tensors='pt', padding='max_length', 
                                  truncation=True, max_length=self.max_length).to(device)
        with torch.no_grad():
            outputs = self.bertModel(**tokens)
            lastLayerEMbeddings = outputs.last_hidden_state
        
        positions = torch.arange(0, self.max_length, device=lastLayerEMbeddings.device).unsqueeze(0).expand(batchSize, -1)
        positionalEmbeddings = self.positionalEmbedding(positions)
        
        textEmbeddings = self.textProjection(lastLayerEMbeddings)
        textEmbeddings = textEmbeddings + positionalEmbeddings
        textEmbeddings = self.textlayerNorm(textEmbeddings)
        
        temporalPositions = torch.arange(0, self.sequenceLength, device=device).unsqueeze(0).expand(batchSize, -1)
        temporal_queries = self.temporalPositionalEmbedding(temporalPositions)
        
        frame_text_features, _ = self.textMultiAttention(
            query=temporal_queries,
            key=textEmbeddings,
            value=textEmbeddings
        )
        
        causal_mask = torch.triu(torch.ones(self.sequenceLength, self.sequenceLength, device=device), diagonal=1).bool()
        
        decoderOut = self.decoder(
            tgt=frame_text_features, 
            memory=textEmbeddings, 
            tgt_mask=causal_mask
        )
        decoderOut = self.decoder_norm(decoderOut)
        
        encoding_indices = self.predictIndices(decoderOut)
        return encoding_indices
    
# t2v = Text2Video(embedDimension=256, sequenceLength=40, codeBookDim=1024, hiddenLayers=6, heads=8, feedForwardDim=2048, drop=0.15)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# t2v.to(device)

# text = ["a cat jumping on a bed", "A man Walking", "He is Running"]
# logits = t2v(text, device)
# logits.shape


BATCH_SIZE = 2
embedDimension = 256
sequenceLength = 40
codeBookDim = 1024
hiddenLayers=6
heads=8
feedForwardDim=2048
drop=0.15
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tranform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


torchDataset = FrameDataset(dataset, totalSequence=sequenceLength, transform=tranform)
dataloader = DataLoader(torchDataset, batch_size=BATCH_SIZE, shuffle = True, num_workers=8, persistent_workers=True)
model = Text2Video(embedDimension=embedDimension, sequenceLength=sequenceLength, codeBookDim=codeBookDim, hiddenLayers=hiddenLayers, heads=heads, feedForwardDim=feedForwardDim, drop=drop)
model = torch.nn.DataParallel(model)
model.to(device)

lossFn =  nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)#, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


epochs = 1000


start_epoch = 0

checkpoint_path = ""
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
else:
    print("Loading pretrained model...")
    # modelValA = torch.load("./projects/t2v-gif/models/t2VModel.pt", map_location=torch.device('cpu'))
    # model.load_state_dict(modelValA)

model = torch.nn.DataParallel(model)
model.to(device)

for each_epoch in range(start_epoch, epochs):
    model.train()
    decoderLoss = 0.0
    
    loop = tqdm(dataloader, f"{each_epoch}/{epochs}")
    decoderLoss = 0.0
    for X, Y in loop:
        # print(X.shape, Y)
        with torch.no_grad():
            _, _, _, _, encoding_indices, _, _ = modelA(X)
        
        y_pred = model(Y, device)
        # break
        # print(y_pred.shape, encoding_indices.shape)
        y_pred_reshaed = rearrange(y_pred, 'b t d -> (b t) d')
        encoding_indices_flat = rearrange(y_pred, 'b t d -> (b t) d', b = BATCH_SIZE, t = sequenceLength, d = codeBookdim)
        loss = lossFn(y_pred_reshaed, encoding_indices_flat)
        decoderLoss += loss.item()
   
        
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loop.set_postfix({
            "Decoder Loss": f"{decoderLoss}"
        })

    decoderLoss /= len(dataloader)   
    
    torch.save({
        'epoch': each_epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)
    
    
    wandb.log({
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Decoder Loss": decoderLoss
    })
    scheduler.step()
 