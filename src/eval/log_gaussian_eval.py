import numpy as np 
import math 

def eval_log_prob_syn_one( true_f, pred_f, true_thresh = math.log( 0.5 ), num_window_accuracy = 5 ):
    """
    takes in the true predict logn prob and true log prob judge whether the prediction is accurate enough.
    Two criteria:
        1. there is an onset if at least one value of the pred log is above true thresh within prediction window
        2. The onset is labeled as the window with peak log prob, if the pred peak is less than num_window_accuracy away
            from true peak, then the onset prediction is considered correct.

        Args:
            true_f - str.npy: the path where true labels are located
            pred_f - str.npy: the path where pred labels are located
            true_thresh - float: the threshold where a window log prob can be considered as having onset 
            num_window_accuracy - int: number of windows max for a 
    """

    true = np.load( true_f )
    pred = np.load( pred_f )
    assert true.shape == pred.shape
    num_sample, window_size = true.shape

    # filter out if there is an onset
    true_is_onset = ( true >= true_thresh  ) # n * window_size
    true_is_onset = np.any( true_is_onset, axis = 1 ) # ( n, )

    pred_is_onset = ( pred >= true_thresh  ) # n * window_size
    pred_is_onset = np.any( pred_is_onset, axis = 1 ) # ( n, )

    # figure out where the onset is
    true_loc = np.argmax( true, axis = 1 )
    pred_loc = np.argmax( pred, axis = 1 )

    distance = np.abs( true_loc - pred_loc )

def eval_wrapper():
    pass