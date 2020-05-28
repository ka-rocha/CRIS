from cris.data import TableData
from cris.classify import Classifier
from cris.regress import Regressor
from cris.sample import Sampler

# generally useful imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial.distance import cdist

DEFAULT_TABLEDATA_KWARGS = {
    "table_paths": ["../examples/data/synth_data.dat"],
    "input_cols": ["input_1", "input_2"],
    "output_cols": ["class", "output_1"],
    "class_col_name":"class",
    "n_neighbors":[2],
    "my_DataFrame":None,
    "verbose":False,
    }

DEFAULT_CLASSIFICATION_KWARGS = {
    "classifier_names": ["linear"],
    "verbose":False,
    }

DEFAULT_REGRESSION_KWARGS = {
    "regressor_names": ["rbf",],
    "verbose":False,
    }

DEFAULT_SAMPLER_KWARGS = {
    "T_max":50,
    "N_tot":500,
    "init_pos": [0,0],
    "target_dist": "TD_classification",
    "classifier_name": "linear",
    "c_spacing":1.3,
    "alpha": [1.,1.],
    "verbose":True,
    "trace_plots":False,
    "TD_verbose": False,  # target distribution kwarg
    "TD_BETA": 1.0,  # target distribution kwarg
}

DEFAULT_PROPOSE_KWARGS = {
    "kappa":100,
    "shuffle":False,
    "norm_steps":False,
    "add_mvns_together":False,
    "var_mult":None,
    "seed":None,
    "n_repeats":1,
    "max_iters":5e3,
    "verbose":False,
    "pred_classifier_name": "linear",
    }

DEFAULT_ALL_KWARGS = {
    "TableData_kwargs":DEFAULT_TABLEDATA_KWARGS,
    "Classifier_kwargs":DEFAULT_CLASSIFICATION_KWARGS,
    "Regressor_kwargs":DEFAULT_REGRESSION_KWARGS,
    "Sampler_kwargs":DEFAULT_SAMPLER_KWARGS,
    "Propose_kwargs":DEFAULT_PROPOSE_KWARGS,
    }

def get_all_default_kwargs():
    return DEFAULT_ALL_KWARGS

def get_new_query_points( N_new_points=1, TableData_kwargs = {},
                                  Classifier_kwargs={}, Regressor_kwargs={},
                                  Sampler_kwargs={}, Propose_kwargs={}, threshold=1e-5 ):
    """Run the cris algorithm to propose new query points to be labeled.

    Parameters
    ----------
    N_new_points : int, optional
        Number of new query points desired.
    TableData_kwargs : dict, optional
        Kwargs used for initializing TableData.
    Classifier_kwargs : dict, optional
        Kwargs used for the Classifier method 'train_everything'.
    Regressor_kwargs : dict, optional
        Kwargs used the Regressor method 'train_everything'.
    Sampler_kwargs : dict, optinal
        Kwargs used for choosing Sampler target distribution and the method 'run_PTMCMC'.
    Propose_kwargs : dict, optional
        Kwargs used in the Sampler method 'get_proposed_points' and the Classifier
        method 'get_class_predictions'.

    Returns
    -------
    proposed_points : ndarray
        Now query points.
    pred_class : array
        For all proposed points, the best prediction from the trained classifier.
    """

    # TableData
    table_obj = TableData(**TableData_kwargs)

    # Classifier
    cls_obj = Classifier(table_obj)
    cls_obj.train_everything(**Classifier_kwargs)

    # Regressor
    if Regressor_kwargs is not None:
        regr_obj = Regressor(table_obj)
        regr_obj.train_everything(**Regressor_kwargs)
    else:
        regr_obj = None

    # Sampler
    sampler_obj = Sampler( classifier = cls_obj, regressor = regr_obj )

    target_dist_name = Sampler_kwargs.get("target_dist", "TD_classification")
    target_dist_obj =  eval( "sampler_obj.{0}".format(target_dist_name) )
    Sampler_kwargs['target_dist'] = target_dist_obj

    chain_step_history, T_list = sampler_obj.run_PTMCMC(**Sampler_kwargs)
    last_chain_hist = chain_step_history[len(T_list)-1]

    # burn in - default to use entire chain
    where_to_cut = int( len(last_chain_hist) * Propose_kwargs.get("cut_fraction", 0) )
    last_chain_hist = last_chain_hist[where_to_cut:]

    # propose new points
    classifier_name = Propose_kwargs.get("pred_classifier_name", "rbf")
    if classifier_name not in Classifier_kwargs.get('classifier_names'):
        raise Exception("Predictions must be with a trained classifier. '{0}' was given".format(classifier_name) )
    if N_new_points == 1:
        proposed_points = last_chain_hist[-1] # take last point in chain
    else:
        kappa = Propose_kwargs.get("kappa",150)
        proposed_points, final_kappa = sampler_obj.get_proposed_points(last_chain_hist, N_new_points, kappa,
                                                                 **Propose_kwargs)
    pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions(classifier_name, proposed_points, return_ids=False)
    return proposed_points, pred_class


def check_dist( original, proposed, threshold = 1e-5):
    """Checks euclidean distance between the original and proposed points.
    Proposed points a distance >= threshold are accepted.

    Parameters
    ----------
    original : ndarray
        Original points previously run.
    proposed : ndarray
        Proposed points for new simulations.
    threshold : float, optional
        The theshold distance between acceptance and rejection.

    Returns
    -------
    proposed_above_thresh_for_all_original : bool, array
        True if the distance between the proposed point is >= threshold.

    Notes
    -----
    The purpose of this function is to not propose points that are some
    threshold away from already accepted points.
    """
    distances = cdist(proposed, original, 'euclidean')
    above_thresh = distances >= threshold
    proposed_above_thresh_for_all_original = [i.all() for i in above_thresh]
    return np.array(proposed_above_thresh_for_all_original)


def get_regular_grid_df(N, jitter=False,verbose=False):
    """Given N total points, produce an even grid with
    approximately the same number of evenly spaced points sampled
    from the analytic data set.

    Parameters
    ----------
    N : int
        Total number of points to make into a 2D even grid
    jitter : bool, optional
        Place the center of the grid randomly around (0,0) in the range of one
        bin width while keeping the span in each axis at 6.
    verbose : bool, optional
        Print some diagnostics.

    Returns
    -------
    extra_points : pandas DataFrame
        DataFrame of true data drawn from the analytic classification
        and regression functions.
    """
    root_of_N = np.sqrt(N)
    x_res = int(root_of_N);  y_res = int(root_of_N)
    if verbose:
        print("x_res: {0}, y_res: {1}\nx*y:{2}  |  N:{3}".format(x_res, y_res, x_res*y_res, N ) )
    if jitter:
        bin_width = 6 / x_res  # span / num bins
        random_center = np.random.uniform( low=(-0.5), high=(0.5), size=(2) )
        center_x = bin_width*random_center[0]; center_y = bin_width*random_center[1]
    else:
        center_x = 0; center_y = 0
    X, Y = np.meshgrid( np.linspace(-3+center_x,3+center_x,x_res),
                        np.linspace(-3+center_y,3+center_y,y_res) )
    stacked_points = np.concatenate( (np.vstack(X.flatten()), np.vstack(Y.flatten())), axis=1  )

    from cris.examples.data.analytic_class_regr import get_output
    class_result, regr_output = get_output( *stacked_points.T )

    extra_points = pd.DataFrame()
    extra_points["input_1"] = stacked_points.T[0]
    extra_points["input_2"] = stacked_points.T[1]
    extra_points["class"] = class_result
    extra_points["output_1"] = regr_output
    return extra_points


def get_random_grid_df(N):
    """Given N total points, produce a randomly sampled grid drawn
    from the analytic data set.

    Parameters
    ----------
    N : int
        Total number of points to drawn from a 2D random data set

    Returns
    -------
    random_df : pandas DataFrame
        DataFrame of true data drawn from the analytic classification
        and regression functions.
    """
    stacked_points = np.random.uniform( low=(-3,-3), high=(3,3), size = (N,2) )

    from cris.examples.data.analytic_class_regr import get_output
    class_result, regr_output = get_output( *stacked_points.T )

    random_df = pd.DataFrame()
    random_df["input_1"] = stacked_points.T[0]
    random_df["input_2"] = stacked_points.T[1]
    random_df["class"] = class_result
    random_df["output_1"] = regr_output
    return random_df
