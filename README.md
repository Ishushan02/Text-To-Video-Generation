# Text-To-Video-Generation
This project focuses on developing a Text-to-Video Generation Model trained on the Tumblr-GIF dataset.

Reading the Data

Process of Creating Model

Part1 - Training Frames on VQVAE from (https://github.com/Ishushan02/Video-Generation-Flowing-MNIST/blob/main/model-4-VQVAE.ipynb)
Part2 - Train a Auto Regressive Transformer Model combining (positional Embedding, Text Embedding ) -> Giving output an Frame Embedding  throught time space to generate Video Frames..


Text Encoding 
CLIP Based only 77 MAX Tokens as input
Using BERT - 512 MAX .. 
Roberta, GPT all are very much higher..

Model
Added postional embed to text, temporal embed for videos and Encoder in and out
