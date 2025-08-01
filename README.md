# Text-To-Video-Generation
This project focuses on developing a Text-to-Video Generation Model trained on the Tumblr-GIF dataset.

Reading the Data

Process of Creating Model

Part1 - Training Frames on VQVAE from (https://github.com/Ishushan02/Video-Generation-Flowing-MNIST/blob/main/model-4-VQVAE.ipynb)

- 1 issue while training, I am mostly getting black and white output, because MSE(reconstruction) is acurately trying to predict exact structure therefore leaving aside the luminance dominance. So, correcting it by adding perceptual Loss and color Loss(this to handle RGB gradients) 

MAIN REASON

```
+--------+--------------+---------------------+-------------------------------+
| Input  | Ground Truth | MSE Only Output     | MSE + Color + Perceptual     |
+--------+--------------+---------------------+-------------------------------+
| ![IMG] | 🟩🟥🟦         | 🟩⬜⬜ (washed out)   | 🟩🟥🟦 (true color restored)     |
+--------+--------------+---------------------+-------------------------------+
```


- The VQVAE Encoder and Decoder model is not that complex, to store and get all features (coloured) in it spaces, so
adding few blocks of it and also skip connection along with it.


Part2 - Train a Auto Regressive Transformer Model combining (positional Embedding, Text Embedding ) -> Giving output an Frame Embedding  throught time space to generate Video Frames..



Text Encoding 
CLIP Based only 77 MAX Tokens as input
Using BERT - 512 MAX .. 
Roberta, GPT all are very much higher..

Model
Added postional embed to text, temporal embed for videos and Encoder in and out
