## some information of data:
# electroencephalogram (EEG):
''' 
In an EEG dataset, channels refer to the individual electrodes that are placed on the scalp to measure the electrical activity of the brain. Each channel records the voltage difference between the electrode and a reference electrode, which provides information about the neural activity in the underlying brain region. EEG signals are typically recorded from multiple channels to capture the activity from different regions of the brain.

The number of channels in an EEG dataset depends on the specific recording setup and the number of electrodes used. For example, a typical EEG recording setup may use 32, 64, or 128 electrodes placed on specific locations on the scalp.

Timesteps in an EEG dataset refer to the time interval between successive voltage measurements recorded by each channel. EEG signals are time-varying and change rapidly over time, so it is important to capture the temporal dynamics of the signal. Timesteps are usually measured in milliseconds (ms), and the duration of the recording depends on the experimental design and the specific research question.

In summary, channels in an EEG dataset refer to the individual electrodes used to measure the electrical activity of the brain, while timesteps refer to the time interval between successive voltage measurements recorded by each electrode. Together, channels and timesteps define the shape of the EEG dataset, which is typically a 2D array of voltage values measured across multiple channels and time intervals.
'''


import numpy as np
import pdb

## paths
base_path = '/home/brainimage/'
train_path = 'brain2image/dataset/eeg_imagenet40_cvpr_2017/train/'
eeg_data = 'n02106662_1152_261.npy'

## read numpy file
loaded_array = np.load(base_path + train_path + eeg_data, allow_pickle=True)

print(loaded_array)
print(loaded_array[0].shape) 
# this is my image dataset

print(loaded_array[1].shape) 
# Save the array to a text file
np.savetxt('array.txt', loaded_array[1], fmt='%d')
# this is my eeg dataset
'''
The shape (128, 440) indicates that each sample in your dataset is a 2D array with 128 rows and 440 columns. In other words, your input data consists of 128 channels of EEG signals, each with 440 time points.
'''

# print(loaded_array[2].shape)
# print(loaded_array[3].shape)

# print(loaded_array[4].shape)






