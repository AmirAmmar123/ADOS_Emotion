import numpy as np
def extract_features_for_all_aus(self, aus_pairs):
        features = []
        liveliness = 0
        for au in aus_pairs:
            au_features, new_liveliness = self.extract_aus_features(au[0], au[1])
            features.extend(au_features)
            liveliness += new_liveliness
        features.append(liveliness)
        return features

@staticmethod
def extract_aus_features(au_binary_list, au_intensity_list):
    # au_binary_list, au_intensity_list = args
    min_num_of_ones = 5
    max_consecutive_zeros = 2
    max_percentage_of_zeros = 0.5

    phases_indices = []
    counter = 0
    while counter < len(au_binary_list) - min_num_of_ones:
        # find the beginning of a phase
        if sum(au_binary_list[counter:counter + min_num_of_ones]) == min_num_of_ones:
            start_ind = counter
            num_of_ones = min_num_of_ones
            total_num_of_zeros = 0
            num_of_consecutive_zeros = 0
            phase_len = min_num_of_ones

            is_phase_valid = True
            counter += min_num_of_ones
            while is_phase_valid and counter < len(au_binary_list):
                if au_binary_list[counter] == 1:
                    num_of_consecutive_zeros = 0
                    num_of_ones += 1
                    phase_len += 1
                    counter += 1
                elif num_of_consecutive_zeros + 1 <= max_consecutive_zeros and (
                        total_num_of_zeros + 1) / phase_len <= max_percentage_of_zeros:
                    num_of_consecutive_zeros += 1
                    total_num_of_zeros += 1
                    phase_len += 1
                    counter += 1
                else:
                    # phase has ended
                    phases_indices.append((start_ind, counter - 1))
                    is_phase_valid = False

        counter += 1

    # calculate features
    phases_lens = []
    phases_avg_intensities = []
    for start_ind, end_ind in phases_indices:
        phases_lens.append(end_ind - start_ind + 1)
        phases_avg_intensities.append(np.mean(au_intensity_list[start_ind:end_ind + 1]))

    phases_lens = np.array(phases_lens)

    num_of_phases = phases_lens.size
    len_avg = np.mean(phases_lens)
    len_std = np.std(phases_lens)
    intensity_avg = np.average(phases_avg_intensities, weights=phases_lens)
    intensity_std = np.std(phases_avg_intensities)

    au_liveliness = np.sum(phases_lens)

    return [len_avg, len_std, intensity_avg, intensity_std, num_of_phases], au_liveliness