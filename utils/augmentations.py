import os
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter
from scipy.fftpack import fft, ifft, fftshift

class Augmenter(object):
    """
    It applies a series of semantically preserving augmentations to batch of sequences, and updates their mask accordingly.
    Available augmentations are:
        - History cutout
        - History crop
        - Gaussian noise
        - Spatial dropout
        - Time Wrap
        - random_fourier_transform
        - bandstop_filter
        - harmonic_distortion
        - scramble_phase
        - pitch_shifting
    """
    def __init__(self, cutout_length=4, cutout_prob=0.5, crop_min_history=0.5, 
                       crop_prob=0.5, gaussian_std=0.1, dropout_prob=0.1, 
                       bandstop_prob=0.2, sigma=0.1, knot=4, lowcut=2, 
                       highcut=6, shift=2, harmonic=2, har_amp=0.2, har_prob=0.1, 
                       scrambel_prob=0.1, pitch_prob=0.1, wrap_prob=0.1,
                       is_cuda=True):
        self.cutout_length = cutout_length
        self.cutout_prob = cutout_prob
        self.crop_min_history = crop_min_history
        self.crop_prob = crop_prob
        self.gaussian_std = gaussian_std
        self.dropout_prob = dropout_prob
        self.bandstop_prob = bandstop_prob
        self.wrap_prob = wrap_prob
        self.sigma = sigma
        self.knot = knot
        self.lowcut = lowcut    # For Spectral Band-Pass Filtering
        self.highcut = highcut  # For Spectral Band-Pass Filtering
        self.harmonic = harmonic # For harmonic Distortion
        self.har_prob = har_prob # For harmonic Distortion
        self.har_amp = har_amp
        self.shift = shift      # For Pitch Shifting
        self.pitch_prob = pitch_prob      # For Pitch Shifting
        self.scramble_prob = scrambel_prob # For Phase Scramble
        self.is_cuda = is_cuda
        
        self.augmentations = [self.history_cutout,
                              self.history_crop,
                              self.gaussian_noise,
                              self.spatial_dropout,
                              self.time_wrap]
        
    def __call__(self, sequence, sequence_mask):
        for f in self.augmentations:
            sequence, sequence_mask = f(sequence, sequence_mask)
            
        return sequence, sequence_mask
        
    def history_cutout(self, sequence, sequence_mask):
        
        """
        Mask out some time-window in history (i.e. excluding last time step)
        """
        n_seq, n_len, n_channel = sequence.shape

        #Randomly draw the beginning of cutout
        cutout_start_index = torch.randint(low=0, high=n_len-self.cutout_length, size=(n_seq,1)).expand(-1,n_len)
        cutout_end_index = cutout_start_index + self.cutout_length

        #Based on start and end index of cutout, defined the cutout mask 
        indices_tensor = torch.arange(n_len).repeat(n_seq,1)
        mask_pre = indices_tensor < cutout_start_index
        mask_post = indices_tensor >= cutout_end_index

        mask_cutout = mask_pre + mask_post

        #Expand it through the dimension of channels
        mask_cutout = mask_cutout.unsqueeze(dim=-1).expand(-1,-1,n_channel).long()

        #Probabilistically apply the cutoff to each sequence
        cutout_selection = (torch.rand(n_seq) < self.cutout_prob).long().reshape(-1,1,1)
        
        #If cuda is enabled, we will transfer the generated tensors to cuda
        if self.is_cuda:
            cutout_selection = cutout_selection.cuda()
            mask_cutout = mask_cutout.cuda()

        #Based on mask_cutout and cutout_selection, apply mask to the sequence
        sequence_cutout = sequence * (1-cutout_selection) + sequence * cutout_selection * mask_cutout

        #Update the mask as well 
        sequence_mask_cutout = sequence_mask * (1-cutout_selection) + sequence_mask * cutout_selection * mask_cutout

        return sequence_cutout, sequence_mask_cutout 
        
    def history_crop(self, sequence, sequence_mask):
        """
        Crop the certain window of history from the beginning. 
        """
        
        n_seq, n_len, n_channel = sequence.shape
    
        #Get number of measurements non-padded for each sequence and time step
        nonpadded = sequence_mask.sum(dim=-1).cpu()
        first_nonpadded = self.get_first_nonzero(nonpadded).reshape(-1,1)/n_len #normalized by length

        #Randomly draw the beginning of crop
        crop_start_index = torch.rand(size=(n_seq,1)) 

        #Adjust the start_index based on first N-padded time steps
        # For instance: if you remove first half of history, then this code removes 
        # the first half of the NON-PADDED history.
        crop_start_index = (crop_start_index * (1 - first_nonpadded) * self.crop_min_history + first_nonpadded)
        crop_start_index = (crop_start_index * n_len).long().expand(-1,n_len)  

        #Based on start index of crop, defined the crop mask 
        indices_tensor = torch.arange(n_len).repeat(n_seq,1)
        mask_crop = indices_tensor >= crop_start_index

        #Expand it through the dimension of channels
        mask_crop = mask_crop.unsqueeze(dim=-1).expand(-1,-1,n_channel).long()

        #Probabilistically apply the crop to each sequence
        crop_selection = (torch.rand(n_seq) < self.crop_prob).long().reshape(-1,1,1)
        
        #If cuda is enabled, we will transfer the generated tensors to cuda
        if self.is_cuda:
            crop_selection = crop_selection.cuda()
            mask_crop = mask_crop.cuda()

        #Based on mask_crop and crop_selection, apply mask to the sequence
        sequence_crop = sequence * (1-crop_selection) + sequence * crop_selection * mask_crop

        #Update the mask as well 
        sequence_mask_crop = sequence_mask * (1-crop_selection) + sequence_mask * crop_selection * mask_crop

        return sequence_crop, sequence_mask_crop
        
    def gaussian_noise(self, sequence, sequence_mask):
        """
        Add Gaussian noise to non-padded measurments
        """
        
        #Add gaussian noise to the measurements
    
        #For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()
        #Calculate the noise for all entries
        noise = nn.init.trunc_normal_(torch.empty_like(sequence),std=self.gaussian_std, a=-2*self.gaussian_std, b=2*self.gaussian_std)

        #Add noise only to nonpadded entries
        sequence_noisy = sequence + padding_mask * noise

        return sequence_noisy, sequence_mask
        
    def spatial_dropout(self, sequence, sequence_mask):
        """
        Drop some channels/measurements completely at random.
        """
        n_seq, n_len, n_channel = sequence.shape

        dropout_selection = (torch.rand((n_seq,1,n_channel)) > self.dropout_prob).long().expand(-1,n_len,-1)
        
        #If cuda is enabled, we will transfer the generated tensors to cuda
        if self.is_cuda:
            dropout_selection = dropout_selection.cuda()

        sequence_dropout = sequence * dropout_selection

        sequence_mask_dropout = sequence_mask * dropout_selection

        return sequence_dropout, sequence_mask_dropout

    def time_wrap(self, sequence, sequence_mask):
        """
        Time warp operation for each sequence.
        """
        def interp1d(x, y, x_new):
            x_new = torch.clamp(x_new, x.min(), x.max())
            x = x.contiguous()
            x_new = x_new.contiguous()
            idx = torch.searchsorted(x, x_new)
            idx = torch.clamp(idx, 1, x.shape[0]-1)
            denominator = (x[idx] - x[idx-1])
            # Set denominator to a small value instead of zero
            denominator[denominator == 0] = 1e-8
            frac = (x_new - x[idx-1]) / denominator
            y_new = y[idx-1] * (1 - frac) + y[idx] * frac
            return y_new

        n_seq, n_len, n_channel = sequence.shape

        warp = torch.normal(mean=1.0, std=self.sigma, size=(n_seq, self.knot + 2, n_channel))
        warp_steps = torch.linspace(0, n_len - 1, self.knot + 2).unsqueeze(-1).repeat(n_seq, 1, n_channel)
        orig_steps = torch.linspace(0, n_len - 1, n_len).unsqueeze(-1).repeat(n_seq, 1, n_channel)

        if self.is_cuda:
            warp = warp.cuda()
            warp_steps = warp_steps.cuda()
            orig_steps = orig_steps.cuda()

        # For padded entries, we won't apply warp
        padding_mask = (sequence_mask != 0).long()

        warp_selection = (torch.rand((n_seq, 1, n_channel)) < self.wrap_prob).long().expand(-1, n_len, -1)

        sequence_warp = torch.zeros_like(sequence)
        for i in range(n_seq):
            for dim in range(n_channel):
                if warp_selection[i, 0, dim] == 1:
                    time_warp = torch.cumsum(interp1d(warp_steps[i, :, dim], warp[i, :, dim], orig_steps[i, :, dim]), dim=0)
                    scale = (n_len - 1) / time_warp[-1]
                    warped_seq = interp1d(scale * time_warp, sequence[i, :, dim], orig_steps[i, :, dim])
                    sequence_warp[i, :, dim] = padding_mask[i, :, dim] * warped_seq
                else:
                    sequence_warp[i, :, dim] = sequence[i, :, dim]

        return sequence_warp, sequence_mask

    def RobustTAD(self, sequence, sequence_mask):
        """
        Randomly alter the Fourier transform of the non-padded parts of the sequence
        by adding Gaussian noise to both amplitude and phase spectrum
        """
        n_seq, n_len, n_channel = sequence.shape

        # For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()

        # Apply a random Fourier transform to each sequence
        sequence_transformed = torch.zeros_like(sequence)
        if self.is_cuda:
            sequence_transformed = sequence_transformed.cuda()

        for i in range(n_seq):
            for j in range(n_channel):
                # perform FFT
                transformed_seq = torch.fft.fft(sequence[i,:,j])

                # extract amplitude and phase
                amplitude = torch.abs(transformed_seq)
                phase = torch.angle(transformed_seq)

                # add Gaussian noise to amplitude and phase
                amplitude_noise = torch.normal(mean=0., std=self.sigma, size=amplitude.size(), device=sequence.device)
                phase_noise = torch.normal(mean=0., std=self.sigma, size=phase.size(), device=sequence.device)
                amplitude = amplitude + amplitude_noise
                phase = phase + phase_noise

                # create a new FFT with the modified amplitude and phase
                transformed_seq = amplitude * torch.exp(1j*phase)

                # transform back to time domain
                transformed_seq = torch.real(torch.fft.ifft(transformed_seq))

                # Apply the transformation only to nonpadded entries
                sequence_transformed[i,:,j] = padding_mask[i,:,j] * transformed_seq
                    
        return sequence_transformed, sequence_mask

        
    def random_fourier_transform(self, sequence, sequence_mask):
        """
        Randomly alter the Fourier transform of the non-padded parts of the sequence
        """
        n_seq, n_len, n_channel = sequence.shape

        # For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()

        # Apply a random Fourier transform to each sequence
        sequence_transformed = torch.zeros_like(sequence)
        if self.is_cuda==True:
            sequence_transformed = sequence_transformed.cuda()

        for i in range(n_seq):
            for j in range(n_channel):
                transformed_seq = torch.real(torch.fft.ifft(torch.fft.fft(sequence[i,:,j]) + torch.rand(n_len, device=sequence.device)))
                
                #Apply the transformation only to nonpadded entries
                sequence_transformed[i,:,j] = padding_mask[i,:,j] * transformed_seq
                
        return sequence_transformed, sequence_mask

    

    def bandstop_filter(self, sequence, sequence_mask):
        """
        Apply random bandstop filtering in the frequency domain to the sequence
        """
        n_seq, n_len, n_channel = sequence.shape

        # For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()

        sequence_filtered = torch.zeros_like(sequence)
        if self.is_cuda==True:
            sequence_filtered = sequence_filtered.cuda()

        fs = 20.0  # Sampling frequency
        freq_band = 2  # Length of frequency band to cutout

        for i in range(n_seq):
            for j in range(n_channel):
                if torch.rand(1).item() < self.bandstop_prob:  # Check if we apply the bandstop filter
                    # Convert to frequency domain
                    fft_seq = torch.fft.fft(sequence[i,:,j])

                    # Randomly choose a starting frequency for the bandstop
                    lowcut = torch.randint(0, int(fs/2) - freq_band, (1,)).item()  # Ensure the band fits within the nyquist limit
                    highcut = lowcut + freq_band  # End frequency for the bandstop
                    
                    # Calculate corresponding index for cutoff frequency
                    low_idx = int(n_len * lowcut / fs)
                    high_idx = int(n_len * highcut / fs)

                    # Zero out the specific frequency content
                    fft_seq[low_idx:high_idx] = 0
                    if n_len % 2 == 0:  # For even length sequences, zero out the symmetric part as well
                        fft_seq[-high_idx:-low_idx] = 0
                    else:  # For odd length sequences
                        fft_seq[-high_idx+1:-low_idx+1] = 0

                    # Convert back to time domain
                    filtered_seq = torch.real(torch.fft.ifft(fft_seq))

                    # Apply the filtering only to nonpadded entries
                    sequence_filtered[i,:,j] = padding_mask[i,:,j] * filtered_seq
                else:
                    # If we're not applying the filter, just keep the original sequence
                    sequence_filtered[i,:,j] = sequence[i,:,j]

        return sequence_filtered, sequence_mask



    
    def harmonic_distortion(self, sequence, sequence_mask):
        """
        Add harmonic distortion to the sequence
        """
        n_seq, n_len, n_channel = sequence.shape

        # For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()

        sequence_distorted = torch.zeros_like(sequence)
        if self.is_cuda:
            sequence_distorted = sequence_distorted.cuda()
            padding_mask = padding_mask.cuda()

        har_selection = (torch.rand((n_seq, 1, n_channel)) < self.har_prob).long().expand(-1, n_len, -1)

        for i in range(n_seq):
            for j in range(n_channel):
                if har_selection[i, 0, j] == 1:
                    distorted_seq = sequence[i, :, j] + self.har_amp * torch.sin(self.harmonic * torch.arange(n_len, device=sequence.device).float() * 2.0 * np.pi / n_len)
                    sequence_distorted[i, :, j] = padding_mask[i, :, j] * distorted_seq
                else:
                    sequence_distorted[i, :, j] = sequence[i, :, j]

        return sequence_distorted, sequence_mask



    def scramble_phase(self, sequence, sequence_mask):
        """
        Apply phase scrambling to the sequence
        """
        n_seq, n_len, n_channel = sequence.shape

        # For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()

        sequence_scrambled = torch.zeros_like(sequence)
        if self.is_cuda:
            sequence_scrambled = sequence_scrambled.cuda()

        scramble_selection = (torch.rand((n_seq, 1, n_channel)) < self.scramble_prob).long().expand(-1, n_len, -1)

        for i in range(n_seq):
            for j in range(n_channel):
                if scramble_selection[i, 0, j] == 1:
                    scrambled_seq = self.scramble_phase_per_channel(sequence[i, :, j])
                    sequence_scrambled[i, :, j] = padding_mask[i, :, j] * scrambled_seq
                else:
                    sequence_scrambled[i, :, j] = sequence[i, :, j]

        return sequence_scrambled, sequence_mask

    def scramble_phase_per_channel(self, data):
        """
        Helper function to apply phase scrambling
        """
        # Transform to frequency domain
        fft_data = torch.fft.fft(data)

        # Calculate magnitude and phase of fft_data
        magnitude = torch.sqrt(fft_data.real**2 + fft_data.imag**2)
        phase = torch.atan2(fft_data.imag, fft_data.real)

        # Randomly scramble the phase
        random_phases = 2 * np.pi * torch.rand_like(phase) - np.pi
        phase = phase + random_phases  # Add random phase to the original phase

        # Convert magnitude and phase back to real and imag
        fft_data_scrambled = torch.view_as_complex(torch.stack([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=-1))

        # Transform back to time domain
        y = torch.real(torch.fft.ifft(fft_data_scrambled))
        return y


    def pitch_shifting(self, sequence, sequence_mask):
        """
        Apply pitch shifting to the sequence
        """
        n_seq, n_len, n_channel = sequence.shape

        # For padded entries, we won't add noise
        padding_mask = (sequence_mask != 0).long()

        sequence_shifted = torch.zeros_like(sequence)
        if self.is_cuda:
            sequence_shifted = sequence_shifted.cuda()

        pitch_selection = (torch.rand((n_seq, 1, n_channel)) < self.pitch_prob).long().expand(-1, n_len, -1)

        for i in range(n_seq):
            for j in range(n_channel):
                if pitch_selection[i, 0, j] == 1:
                    shifted_seq = self.pitch_shift_per_channel(sequence[i, :, j], self.shift)
                    sequence_shifted[i, :, j] = padding_mask[i, :, j] * shifted_seq
                else:
                    sequence_shifted[i, :, j] = sequence[i, :, j]

        return sequence_shifted, sequence_mask


    def pitch_shift_per_channel(self, data, shift):
        """
        Helper function to apply pitch shifting
        """
        # Pad data to nearest power of 2
        data_padded = self.pad_to_nearest_power_of_2(data)

        # Transform to frequency domain
        X = torch.fft.fft(data_padded)

        # Shift the frequencies
        X_shifted = torch.fft.fftshift(X)
        X_shifted = torch.roll(X_shifted, int(len(data_padded) // shift))
        X_shifted = torch.fft.fftshift(X_shifted)

        # Transform back to time domain
        y = torch.real(torch.fft.ifft(X_shifted))

        # Unpad back to original size
        y = y[:len(data)]
        
        return y


    def pad_to_nearest_power_of_2(self, x):
        """
        Pad a 1D tensor to the nearest power of 2.
        """
        original_len = x.shape[0]
        target_len = 2**torch.ceil(torch.log2(torch.tensor(original_len, dtype=torch.float32))).long()

        padded_x = torch.nn.functional.pad(x, (0, target_len - original_len))

        return padded_x


    def get_first_nonzero(self, tensor2d):
        """
        Helper function to get the first nonzero index for the 2nd dimension
        """
        
        nonzero = tensor2d != 0
        cumsum = nonzero.cumsum(dim=-1)

        nonzero_idx = ((cumsum == 1) & nonzero).max(dim=-1).indices

        return nonzero_idx



def concat_mask(seq, seq_mask, use_mask=False):
    if use_mask:
        seq = torch.cat([seq, seq_mask], dim=2)
    return seq
