# EEGStyleGAN-ADA
Pytorch code of paper "Learning Robust Deep Visual Representations from EEG Brain Recordings."

1. Anaconda environment yml file is present in the Anaconda folder. Use it to create a conda environment.
2. For StyleGAN-ADA, we have used the official Pytorch implementation. [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch)
3. Some network weights are not added due to file size restrictions on GitHub, and it's impossible to add them here without breaking anonymity so we will release it later.
4. Command to train the GAN is mentioned in the Txt file.

## Introduction

Decoding the human brain has been a hallmark of neuroscientists and Artificial Intelligence researchers alike. Reconstruction of Visual images from brain Electroencephalography (EEG) signals has garnered a lot of interest due to its applications in brain-computer interfacing. This study proposes a two-stage method where the first step is to obtain EEG-derived features for robust learning of deep representations and subsequently utilize the learned representation for image generation and classification. We demonstrate the generalizability of our feature extraction pipeline across three different datasets using deep-learning architectures with supervised and contrastive learning methods. We have performed the zero-shot EEG classification task to support the generalizability claim further. We observed that a subject invariant linearly separable visual representation was learned using EEG data alone in a unimodal setting that gives better k-means accuracy as compared to a joint representation learning between EEG and images. Finally, we proposed a novel framework to transform unseen images into the EEG space and reconstruct them with approximation, showcasing the potential for image reconstruction from EEG signals. Our proposed image synthesis method from EEG shows 62.9% and 36.13% inception score improvement on the EEGCVPR40 and the Thoughtviz datasets, which is better than state-of-the-art performance in GAN.

| Feature Extraction and Image Synthesis Architecture  |
|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/architecture.png" width="1024px" height="512px"/>  |

| Learned EEG Space using Triplet Loss with LSTM and CNN Architecture  |
|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/eegspace.png" width="1024px" height="512px"/>  |

| CVPR40 Dataset (40 Classes)  | ThoughtViz Dataset (10 Classes) |
|---|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/seed0000-min.png" width="512px" height="512px"/>  | <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/fakes005725-min.png" width="512px" height="512px"/>  |
