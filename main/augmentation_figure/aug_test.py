import sys
sys.path.append("../../")
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils.dataset import get_dataset
from utils.augmentations import Augmenter
import torch

from argparse import ArgumentParser

"""
command: 
python aug_test.py --path_src ../../processed_datasets/WISDM --id_src 7 --path_trg ../../processed_datasets/WISDM --id_trg 2
"""


def plot_data(sequence, aug_sequence, save_path):

    plt.rcParams['font.family'] = "monospace"
    color_dict = {
        0 : "coral",
        1 : "mediumslateblue",
        2 : "hotpink",
        "teacher": "lawngreen",
        "klla": "cornflowerblue"
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))
    ax1.set_title('Data')
    ax2.set_title('Augmented Data')
    sequence = sequence.permute(0, 2, 1)
    aug_sequence = aug_sequence.permute(0, 2, 1)
    id_list = [i for i in range(len(sequence[0][0]))]
    ax2.plot(id_list, aug_sequence[0][0], c=color_dict[0], label='x')
    ax2.plot(id_list, aug_sequence[0][1], c=color_dict[1], label='y')
    ax2.plot(id_list, aug_sequence[0][2], c=color_dict[2], label='z')
    ax1.plot(id_list, sequence[0][0], c=color_dict[0], label='x')
    ax1.plot(id_list, sequence[0][1], c=color_dict[1], label='y')
    ax1.plot(id_list, sequence[0][2], c=color_dict[2], label='z')
    plt.subplots_adjust(hspace=0.3)
    ax1.legend(loc="best")
    ax2.legend(loc="best")

    plt.show()
    plt.clf()
    plt.savefig(save_path)


def plot_sequences(sequence_id, sequence, sequence_aug, save_path=''):
    # Create a figure and subplots
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey='row')
    color = ['r', 'g','b']
    # Plot the original sequence channels
    for i in range(3):
        axes[i, 0].plot(sequence[sequence_id, :, i].T,color[i], label=f'Channel {i+1}')
        axes[i, 0].set_title(f'Original Sequence - Channel {i+1}')
        axes[i, 0].legend()

    # Plot the augmented sequence channels
    for i in range(3):
        axes[i, 1].plot(sequence_aug[sequence_id, :, i].T,color[i], label=f'Channel {i+1}')
        axes[i, 1].set_title(f'Augmented Sequence - Channel {i+1}')
        axes[i, 1].legend()

    # Adjust the layout
    fig.tight_layout()

    # Save the figure if save_path is provided
    filename = f'time.png'
    save_filepath = os.path.join(save_path, filename)
    fig.savefig(save_filepath)
    print(f"Figure saved at: {save_filepath}")

    # Close the figure
    plt.close(fig)

    
def plot_channels(sequence_id, sequence, sequence_aug, save_path=''):
    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Iterate over each channel
    for channel in range(sequence.shape[2]):
        # Plot the original sequence as a line in the current subplot
        axs[channel].plot(sequence[sequence_id, :, channel].T, 'r',label='Original')
        # Plot the augmented sequence as a line in the current subplot
        axs[channel].plot(sequence_aug[sequence_id, :, channel].T,'g', label='Augmented')

        # Set the title and labels for the current subplot
        axs[channel].set_title(f'Channel {channel+1}')
        axs[channel].set_xlabel('Time Step')
        axs[channel].set_ylabel('Value')
        axs[channel].legend()

        # Adjust the layout
        # fig.tight_layout()

    # Save the figure if save_path is provided
    filename = f'time.png'
    save_filepath = os.path.join(save_path, filename)
    fig.savefig(save_filepath)
    print(f"Figure saved at: {save_filepath}")

    # Close the figure
    plt.close(fig)


def plot_frequency_domain(sequence_id, sequence_before, sequence_after, sequence_mask, sample_rate, save_path=''):
    n_seq, n_len, n_channel = sequence_before.shape

    # Create a figure and subplots
    fig, axes = plt.subplots(n_channel, 1, figsize=(12, 6*n_channel))

    # Compute frequencies for FFT bins
    freq = np.fft.fftfreq(n_len, d=1./sample_rate)
    # Iterate over each channel
    for j in range(n_channel):
        # Make sure to only consider non-padded parts of the sequence
        sequence_before_np = sequence_before[sequence_id, sequence_mask[sequence_id,:,j]==1, j].cpu().numpy()
        sequence_after_np = sequence_after[sequence_id, sequence_mask[sequence_id,:,j]==1, j].cpu().numpy()

        # Compute FFT for both sequences
        fft_before = np.fft.fft(sequence_before_np)
        fft_after = np.fft.fft(sequence_after_np)
        # Create ordered frequency and FFT arrays
        freq_ordered = np.concatenate([freq[n_len//2:], freq[:n_len//2]])
        fft_before_ordered = np.concatenate([fft_before[n_len//2:], fft_before[:n_len//2]])
        fft_after_ordered = np.concatenate([fft_after[n_len//2:], fft_after[:n_len//2]])


        # Plotting
        ax = axes[j]
        ax.plot(freq_ordered, np.abs(fft_after_ordered), 'r',label='After Augmentation')
        ax.plot(freq_ordered, np.abs(fft_before_ordered),'b', label='Before Augmentation')
        ax.set_title(f'Frequency Domain - Channel {j+1}')
        ax.legend()
        ax.grid(True)

    # Adjust the layout
    # fig.tight_layout()

    # Save the figure if save_path is provided
    filename = f'frequency.png'
    save_filepath = os.path.join(save_path, filename)
    fig.savefig(save_filepath)
    print(f"Figure saved at: {save_filepath}")

    # Close the figure
    plt.close(fig)

def plot_phase_spectrum(sequence_id, sequence_before, sequence_after, sequence_mask, sample_rate, save_path=''):
    n_seq, n_len, n_channel = sequence_before.shape

    # Create a figure and subplots
    fig, axes = plt.subplots(n_channel, 2, figsize=(12, 6*n_channel))  # Change to n_channel rows and 2 columns

    # Compute frequencies for FFT bins
    freq = np.fft.fftfreq(n_len, d=1./sample_rate)

    # Iterate over each channel
    for j in range(n_channel):
        # Make sure to only consider non-padded parts of the sequence
        sequence_before_np = sequence_before[sequence_id, sequence_mask[sequence_id,:,j]==1, j].cpu().numpy()
        sequence_after_np = sequence_after[sequence_id, sequence_mask[sequence_id,:,j]==1, j].cpu().numpy()

        # Compute FFT for both sequences
        fft_before = np.fft.fft(sequence_before_np)
        fft_after = np.fft.fft(sequence_after_np)

        # Create ordered frequency and FFT arrays
        freq_ordered = np.concatenate([freq[n_len//2:], freq[:n_len//2]])
        fft_before_ordered = np.concatenate([fft_before[n_len//2:], fft_before[:n_len//2]])
        fft_after_ordered = np.concatenate([fft_after[n_len//2:], fft_after[:n_len//2]])

        # Plotting
        ax1 = axes[j, 0]  # First column for 'Before Augmentation'
        ax2 = axes[j, 1]  # Second column for 'After Augmentation'

        ax1.plot(freq_ordered, np.angle(fft_before_ordered),'b', label='Before Augmentation')  # Plot 'Before Augmentation'
        ax1.set_title(f'Phase Spectrum - Channel {j+1} (Before Augmentation)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim([-2*np.pi, 2*np.pi])  # Adjust y-axis limits here


        ax2.plot(freq_ordered, np.angle(fft_after_ordered), 'r',label='After Augmentation')  # Plot 'After Augmentation'
        ax2.set_title(f'Phase Spectrum - Channel {j+1} (After Augmentation)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim([-2*np.pi, 2*np.pi])  # Adjust y-axis limits here


    # Adjust the layout
    fig.tight_layout()

    # Save the figure if save_path is provided
    filename = f'phase_spectrum.png'
    save_filepath = os.path.join(save_path, filename)
    fig.savefig(save_filepath)
    print(f"Figure saved at: {save_filepath}")

    # Close the figure
    plt.close(fig)





def main(args):
    # Usage example:
    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    print('dataset shape:', dataset_src.sequence.shape)

    index = 0 # which sequence
    channel_index = 0 # which channel
    sample_rate = 20
    # sequence = dataset_src[index, :, channel_index]
    sequence = dataset_src.sequence
    sequence = torch.from_numpy(sequence)
    aug = Augmenter(is_cuda=False, gaussian_std=0.2 ,cutout_length=20,cutout_prob=1, crop_prob=1,
                    bandstop_prob=1,har_prob=1, dropout_prob=0.5,scrambel_prob=1,
                    pitch_prob=1,wrap_prob=1,sigma=0.5, harmonic=2, har_amp=0.3)
    mask = torch.ones_like(sequence)

    sequence_aug, sequence_mask = aug.gaussian_noise(sequence, mask)
    # sequence_aug, sequence_mask = aug.history_cutout(sequence, mask)
    # sequence_aug, sequence_mask = aug.history_crop(sequence, mask) # from the beginning
    # sequence_aug, sequence_mask = aug.spatial_dropout(sequence, mask)
    
    # sequence_aug, sequence_mask = aug.time_wrap(sequence, mask)


    # sequence_aug, sequence_mask = aug.bandstop_filter(sequence, mask)
    # sequence_aug, sequence_mask = aug.random_fourier_transform(sequence, mask)
    # sequence_aug, sequence_mask = aug.harmonic_distortion(sequence, mask)
    # sequence_aug, sequence_mask = aug.scramble_phase(sequence, mask)
    # sequence_aug, sequence_mask = aug.pitch_shifting(sequence, mask)
    # sequence_aug, sequence_mask = aug.RobustTAD(sequence, mask)
    print('sequence augmented shape', sequence_aug.shape)
    print('sequence mask shape', sequence_mask.shape)

    # plot_frequency_domain(1 ,sequence, sequence_aug, sequence_mask, sample_rate)
    # plot_phase_spectrum(1 ,sequence, sequence_aug, sequence_mask, sample_rate)
    plot_sequences(1 ,sequence, sequence_aug)
    # plot_channels(1 ,sequence, sequence_aug)




        

# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='cluda')

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99) #CLUDA
    parser.add_argument('-qs', '--queue_size', type=int, default=98304) #CLUDA
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true') #CLUDA
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=20)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--n_layers', type=int, default=1) #VRADA
    parser.add_argument('--h_dim', type=int, default=64) #VRADA
    parser.add_argument('--z_dim', type=int, default=64) #VRADA
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64') #All TCN models
    parser.add_argument('--kernel_size_TCN', type=int, default=3) #All TCN models
    parser.add_argument('--dilation_factor_TCN', type=int, default=2) #All TCN models
    parser.add_argument('--stride_TCN', type=int, default=1) #All TCN models
    parser.add_argument('--hidden_dim_MLP', type=int, default=256) #All classifier and discriminators

    #The weight of the domain classification loss
    parser.add_argument('-w_d', '--weight_domain', type=float, default=1.0)
    parser.add_argument('-w_kld', '--weight_KLD', type=float, default=1.0) #VRADA
    parser.add_argument('-w_nll', '--weight_NLL', type=float, default=1.0) #VRADA
    parser.add_argument('-w_cent', '--weight_cent_target', type=float, default=1.0) #CDAN
    #Below weights are defined for CLUDA
    parser.add_argument('--weight_loss_src', type=float, default=1.0)
    parser.add_argument('--weight_loss_trg', type=float, default=1.0)
    parser.add_argument('--weight_loss_ts', type=float, default=1.0)
    parser.add_argument('--weight_loss_disc', type=float, default=1.0)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')

    parser.add_argument('--path_src', type=str, default='../Data/miiv_fullstays')
    parser.add_argument('--path_trg', type=str, default='../Data/aumc_fullstays')
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=int, default=1)
    parser.add_argument('--id_trg', type=int, default=2)

    parser.add_argument('--task', type=str, default='decompensation')

    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    main(args)