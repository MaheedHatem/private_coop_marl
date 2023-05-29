import numpy as np
from math import factorial
from matplotlib import pyplot as plt

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# results_dirs = {
#     'Centralized': ['saved_models\\course_results\\coin_gather_2_CentralizedController_DQNAgent_done'],
#  #   'Decentralized': ['saved_models\\course_results\\coin_gather_2_DecentralizedController_DQNAgent_done'],
#  #   'Proposed β = 0.5, ζ = 0': 'saved_models\\coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.5_done',
#     'Proposed old β = 0.8, ζ = 0': ['saved_models\\course_results\\coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done'],
#     'Proposed β = 0.8, ζ = 0': ['saved_models\\coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8-1',
#                                 'saved_models\\coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8-2',
#                                 'saved_models\\coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8-3']
# }

# results_dirs = {
#     'Proposed Learned estimated Q or Learned Q': ['saved_models\coin_gather_3_DecentralizedController_DQNRewardAgent_0.0_0.8_learnedq_done'],
#     'Proposed Estimator or Learned Q': ['saved_models\coin_gather_3_DecentralizedController_DQNRewardAgent_0.0_0.8_done'],
#     'Proposed Estimator or Learned Q 0.95': ['saved_models\\coin_gather_3_DecentralizedController_DQNRewardAgent_0.0_0.95_done'],
#     'Proposed Estimator + local Q': ['saved_models\\coin_gather_3_DecentralizedController_DQNRewardAgent_0.0_0.8_done_both'],
#     'Decentralized': ['saved_models\coin_gather_3_DecentralizedController_DQNAgent_done'],
#     'Centralized': ['saved_models\coin_gather_3_CentralizedController_DQNAgent_done']

# }

results_dirs = {
    #'Proposed learned q': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done'],
    #'Proposed predictor': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_predictor_done'],
    'Proposed learned q gamma 0.7': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done_target_predict'],
    #'Proposed learned q gamma 0.9': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done_0.9'],
    #'Proposed learned q gamma 0.9 similarity 0': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done_simi_0'],
    'Proposed learned q gamma 0.9 similarity 1': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done_simi_1'],
    'Proposed learned q gamma 0.9 similarity 1-2': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done_simi_1-2']
    #'Proposed learned q gamma 0.9 similarity 2': ['saved_models/coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done_simi_2']

}

# results_dirs = {
#     'Proposed β = 0.8, ζ = 0': 'coin_gather_2_DecentralizedController_DQNRewardAgent_0.0_0.8_done',
#    'Proposed β = 0.8, ζ = 0.2': 'coin_gather_2_DecentralizedController_DQNRewardAgent_0.2_0.8_done',
#    'Proposed β = 0.8, ζ = 0.4' : 'coin_gather_2_DecentralizedController_DQNRewardAgent_0.4_0.8_done',
#    'Proposed β = 0.8, ζ = 0.8' : 'coin_gather_2_DecentralizedController_DQNRewardAgent_0.8_0.8_done'
# }
# results_dirs = {
#     'Centralized': 'coin_gather_3_CentralizedController_DQNAgent_done',
#     'Decentralized': 'coin_gather_3_DecentralizedController_DQNAgent_done',
#     'Proposed β = 0.8, ζ = 0': 'coin_gather_3_DecentralizedController_DQNRewardAgent_0.0_0.8_done'
# }

# results_dirs = {
#     'Centralized': 'coin_gather_3_CentralizedController_DQNAgent',
#     'Decentralized': 'coin_gather_3_DecentralizedController_DQNAgent',
#     'Proposed β = 0.8, ζ = 0': 'coin_gather_3_DecentralizedController_DQNRewardAgent_0.0_0.8'
# }
# results_dirs = {
#     'Centralized': 'coin_gather_5_CentralizedController_DQNAgent_done',
#     'Proposed β = 0.8': 'coin_gather_5_DecentralizedController_DQNRewardAgent_0.0_0.8_done',
# }
scores = {}
ci = {}
steps = []
for label, results_paths in results_dirs.items():
    results = []
    for result_dir in results_paths:
        results.append(np.loadtxt(f"{result_dir}/results.csv", delimiter=','))
    results = np.stack(results, axis=2)
    scores[label] = np.mean(results, axis=2)
    ci[label] = 1.96 * np.std(results, axis = 2)/np.sqrt(len(results))
    if len(steps):
        assert np.all(steps == scores[label][:, 0])
    else:
        steps = scores[label][:, 0]
    
    indices = range(0,len(steps),1)
    #plt.plot(steps[indices], savitzky_golay(scores[label][indices,column], 51, 3), label=label)
    column = 1
    plt.plot(steps[indices], scores[label][indices,column], label=label)
    plt.fill_between(steps[indices], (scores[label][indices,column]-ci[label][indices,column]), (scores[label][indices,column]+ci[label][indices,column]), color='b', alpha=.1)
    plt.xlabel("Training Step")
    plt.ylabel("Total average reward of all agents")
    plt.title("Three agents with four levers, a general setup")
    
plt.legend()
plt.savefig('Figure.png')
plt.show()