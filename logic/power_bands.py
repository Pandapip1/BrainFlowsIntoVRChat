from logic.base_logic import BaseLogic
from constants import BAND_POWERS

import utils

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, NoiseTypes, WaveletTypes, ThresholdTypes
from scipy.signal import butter, filtfilt

import re
import numpy as np

class PwrBands(BaseLogic):
    LEFT = 'Left'
    RIGHT = 'Right'
    AVERAGE = 'Avg'

    def __init__(self, board, window_seconds=2, ema_decay=0.025):
        super().__init__(board)
        
        board_id = board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_names = BoardShim.get_eeg_names(board_id)

        self.window_seconds = window_seconds
        self.max_sample_size = self.sampling_rate * window_seconds

        # sort left and right channels
        eeg_nums = map(lambda eeg_name: int(''.join(re.findall(r'\d+', eeg_name))), eeg_names)
        chan_num_pairs = list(zip(self.eeg_channels, eeg_nums))
        self.left_chans = [eeg_chan for eeg_chan, eeg_num in chan_num_pairs if eeg_num % 2 != 0]
        self.right_chans = [eeg_chan for eeg_chan, eeg_num in chan_num_pairs if eeg_num % 2 == 0]

        # ema smoothing variables
        self.current_dict = {}
        self.ema_decay = ema_decay

        # detect_artifacts settings
        self.f_params = butter(2, 10 / (self.sampling_rate / 2), btype='low')  # 10 Hz lowpass filter
        self.art_thresh = 100  # 100 uV difference is indicative of blink
    
    def detect_artifacts(self, data):
        ## artifact detection inspired by openbci algorithm
        ## https://openbci.com/community/automated-eye-blink-detection-online-2/
        b, a = self.f_params
        
        # lowpass filter to blink range
        filtered = filtfilt(b, a, data)

        # find median and use difference to it to threshold mask
        median = np.median(filtered, axis=1, keepdims=True)
        diff = np.abs(filtered - median)
        mask = diff > self.art_thresh

        # return true if any artifacts detected
        return np.any(mask)
    
    def get_data_dict(self):
        # get current data from board
        data = self.board.get_current_board_data(self.max_sample_size)

        # denoise and filter data
        for eeg_chan in self.eeg_channels:
            DataFilter.perform_wavelet_denoising(data[eeg_chan], WaveletTypes.DB4, 5, threshold=ThresholdTypes.SOFT)
            DataFilter.remove_environmental_noise(data[eeg_chan], self.sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
        
        # check if artifact in window
        has_artifact = self.detect_artifacts(data[self.eeg_channels])

        # calculate band features for left, right, and overall
        left_powers, _ = DataFilter.get_avg_band_powers(data, self.left_chans, self.sampling_rate, True)
        right_powers, _ = DataFilter.get_avg_band_powers(data, self.right_chans, self.sampling_rate, True)
        avg_powers, _ = DataFilter.get_avg_band_powers(data, self.eeg_channels, self.sampling_rate, True)

        # create location dict
        location_dict = {
            PwrBands.LEFT     : left_powers,
            PwrBands.RIGHT    : right_powers,
            PwrBands.AVERAGE  : avg_powers
        }

        # smooth out powers
        location_dict = {loc : self.location_smooth(loc, powers, has_artifact) for loc, powers in location_dict.items()}

        # create power dicts per location
        def make_power_dict(powers):
            return {bp.name : powers[bp] for bp in BAND_POWERS}
        ret_dict = {loc: make_power_dict(powers) for loc, powers in location_dict.items()}

        return ret_dict
    
    def location_smooth(self, loc_name, target_values, has_artifact):
        current_values, old_target_values = self.current_dict.get(loc_name, (None, None))

        # pause target update on artifact window
        if has_artifact and isinstance(old_target_values, np.ndarray):
            target_values = old_target_values

        # ema to target
        if isinstance(current_values, np.ndarray):
            current_values = utils.smooth(current_values, target_values, self.ema_decay)
        else:
            current_values = target_values
            
        self.current_dict[loc_name] = (current_values, target_values)
        return current_values