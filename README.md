# EEGStyleGAN-ADA
Pytorch code of paper "Learning Robust Deep Visual Representations from EEG Brain Recordings."

1. Anaconda environment yml file is present in the Anaconda folder. Use it to create a conda environment.
2. For StyleGAN-ADA, we have used the official Pytorch implementation. [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch)
3. Some network weights are not added due to file size restrictions on GitHub, and it's impossible to add them here without breaking anonymity, so we will release it later.
4. Command to train the GAN is mentioned in the Txt file.

| Learned EEG Space using Triplet Loss with LSTM and CNN Architecture  |
|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/eegspace.png" width="1024px" height="512px"/>  |

| CVPR40 Dataset (40 Classes)  | ThoughtViz Dataset (10 Classes) |
|---|---|
| <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/seed0000-min.png" width="512px" height="512px"/>  | <img src="https://github.com/prajwalsingh/EEGStyleGAN-ADA/blob/main/images/fakes005725-min.png" width="512px" height="512px"/>  |
