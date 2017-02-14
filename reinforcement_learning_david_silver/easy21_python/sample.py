import numpy as np

def get_feature(ranges, value):
    result = []
    for i, f_range in enumerate(ranges):
        if f_range[0] <= value <= f_range[1]:
            result.append(i)
    return result

def featureize(dealer_sum, player_sum, action):
    dealer_ranges = np.array([[1, 4], [4, 7], [7, 10]])
    player_ranges = np.array([[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]])

    dealer_index = get_feature(dealer_ranges, dealer_sum)
    player_index = get_feature(player_ranges, player_sum)

    features = np.zeros((dealer_ranges.shape[0], player_ranges.shape[0], 2))
    for i in dealer_index:
        for j in player_index:
            features[i][j][action] = 1

    return features

f = featureize(4, 7, 1)


pass