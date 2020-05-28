from cris.data import TableData
from cris.classify import Classifier
from cris.regress import Regressor
from cris.sample import Sampler

from cris.examples.data.analytic_class_regr import get_output

# generally useful imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy
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
    target_dist_obj = getattr(sampler_obj, target_dist_name)
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

    kappa = Propose_kwargs.get("kappa",150)
    proposed_points, final_kappa = sampler_obj.get_proposed_points(last_chain_hist, N_new_points, kappa,
                                                                    **Propose_kwargs)
    pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions(classifier_name, proposed_points, return_ids=False)
    return proposed_points, pred_class




################################################################################
# Functions below this point are designed specifically for testing cris so they
# assume things like the data set being used and do not have the flexibility
# present throughout the rest of the code.
################################################################################





def do_dynamic_sampling( N_starting_points=100, N_final_points=400, new_points_per_iter=20,
                        threshold=1e-6, jitter=False, verbose=False, **all_kwargs  ):
    """For a given number of starting and ending points, run the cris algorithm iteratively
    in step sizes of new_points_per_iter. After each iteration, query points are identified
    using the original 2D snythetic data set.

    Parameters
    ----------
    N_starting_points : int
        Number of starting points to being cris iterations on a 2D grid
        sampled from the original 2D synthetic data set.
    N_final_points : int
        Number of points to converge to after iterating with cris.
    new_points_per_iter : int, array-like
        For every iteration, the number of new query points for cris to propose.
    threshold : float
        New query points are ommited from the next iteration if their euclidean
        distance to other data points is less than the threshold.
    jitter : bool
        Default False, for the starting grid jitter about the center randomly in
        the range of +/- the 1/2 the bin width in each dimesnion.
    verbose : bool
        Print useful things.
    all_kwargs : dict
        Dictionary of all_kwargs passed to get_new_query_points defining
        how every part of the cris algorithm is implemented.
    """
    t0 = time.time()

    original_kwargs = copy.deepcopy(all_kwargs) #!!!

    # analytic classification and regression data set
    my_data = get_regular_grid_df( N_starting_points, jitter=jitter )
    all_kwargs["TableData_kwargs"]["my_DataFrame"] = my_data
    dfs_per_iters = [my_data]; preds_per_iter = [None]

    N_total = int(N_final_points - N_starting_points)
    if verbose: print("Sampling {} total points...".format(N_total))
    num_loops = 0; N_sampled_points = 0
    while N_sampled_points < N_total:
        start_time = time.time()
        if isinstance( new_points_per_iter, int ):
            n = new_points_per_iter
            if abs(N_sampled_points - N_total)/n < 1:
                n = N_total - N_sampled_points
        else:
            try:
                n = new_points_per_iter[num_loops]
            except:
                n = N_total - N_sampled_points

        if verbose:
            print("\n\n\tSTART ITER {0}, init_pos = {1}, n = {2}".format(
                    num_loops, all_kwargs["Sampler_kwargs"]["init_pos"], n))

        new_points, cls_preds = get_new_query_points(N_new_points=n, **all_kwargs )
        if new_points.ndim == 1:
            new_points = np.array([new_points])

        # Since we use .pop we must repopulate everything in the dicts
        all_kwargs = copy.deepcopy(original_kwargs) #!!!!

        # Update dicts
        all_kwargs["TableData_kwargs"]["file_path_list"] = None
        all_kwargs["TableData_kwargs"]["my_DataFrame"] = my_data
        random_init_pos = np.random.uniform(low=-2,high=2, size=(2))
        all_kwargs["Sampler_kwargs"]["init_pos"] = random_init_pos

        # Check distances
        old_points = my_data[["input_1", "input_2"]].to_numpy()
        where_good_bools = check_dist( old_points, new_points, threshold=threshold )
        if np.sum(where_good_bools) != len(new_points):
            print("We are getting rid of {} points below thresh this iter.".format(len(new_points)-np.sum(where_good_bools)) )
        new_points = new_points[where_good_bools]

        # Evaluate query points with anayltic dataset
        class_results, regr_out = get_output( *new_points.T )

        data_products = [new_points.T[0], new_points.T[1], class_results, regr_out]
        new_df = pd.DataFrame()
        for i, col_name in enumerate(my_data.keys()):
            new_df[col_name] = data_products[i]

        new_data = my_data.append( new_df, ignore_index=True )
        # Append data to be returned
        dfs_per_iters.append( new_data )
        my_data = new_data.copy()
        # Append data to be returned
        preds_per_iter.append( cls_preds )

        where_to_cut = int(N_starting_points + N_sampled_points)
        fig, subs = plt.subplots(1,1, figsize=(3.5,3.5), dpi=100)
        subs.set_title("SAMPLE {0}".format(num_loops))
        subs.plot( *random_init_pos, '+', markeredgewidth=1.5, color="red", label="init_pos")
        subs.scatter( my_data["input_1"][0:where_to_cut], my_data["input_2"][0:where_to_cut],
                     alpha=0.5, color = "dodgerblue", label="training")
        subs.scatter( my_data["input_1"][where_to_cut:], my_data["input_2"][where_to_cut:],
                     marker = 'x', color = "C2", label="proposed" )
        subs.set_xlabel("{0} new points this iter".format(np.sum(where_good_bools)))
        plt.legend(bbox_to_anchor=[1, 0, 0.22,1])
        plt.show()


        N_sampled_points += len(new_points)
        num_loops += 1
        if verbose:
            end_of_iter_str = "\tEND ITER {0} in {1:.2f}s".format(num_loops-1, time.time()-start_time)
            print( end_of_iter_str + ", N_sampled_points: {}\n\n".format(N_sampled_points) )

    if verbose:
        print( "\n\nDone. Sampled {0} points.".format(len(my_data)-N_starting_points) )
        print( "Total time: {:.2f}s".format(time.time()-t0)  )

    return dfs_per_iters, preds_per_iter


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


def get_regular_grid_df(N=100, jitter=False,verbose=False, N_ppa=None):
    """Given N total points, produce an even grid with
    approximately the same number of evenly spaced points sampled
    from the analytic data set.

    The number of returned grid points is N only if N is a perfect square.
    Otherwise use N_ppa to define number of points per axis.

    Parameters
    ----------
    N : int
        Total number of points to make into a 2D even grid
    jitter : bool, optional
        Place the center of the grid randomly around (0,0) in the range of
        +/- 1/2 bin width while keeping the span in each axis at 6.
    N_ppa : array-like, optional
        Numbers of points per axis. If provided, it overrides N.
    verbose : bool, optional
        Print some diagnostics.

    Returns
    -------
    extra_points : pandas DataFrame
        DataFrame of true data drawn from the analytic classification
        and regression functions.
    """

    if N_ppa is None:
        root_of_N = np.sqrt(N)
        x_res = int(root_of_N);  y_res = int(root_of_N)
    else:
        x_res = int(N_ppa[0]);  y_res = int(N_ppa[1])
    if verbose:
        print("x_res: {0}, y_res: {1}\nx*y:{2}".format(x_res, y_res, x_res*y_res) )
    if jitter:
        bin_width_x = 6 / x_res  # span / num bins
        bin_width_y = 6 / y_res
        random_center = np.random.uniform( low=(-0.5), high=(0.5), size=(2) )
        center_x = bin_width_x*random_center[0]; center_y = bin_width_y*random_center[1]
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


def get_prediction_diffs( training_df, classifier_name = "linear", regressor_name = "linear",
                         N = 400, verbose=False):
    """From a DataFrame of training data, train a classifier and get both
    the predicitons and actual classification in the classification space
    where the analytic function is defined. Also calculate the difference
    between the true regression function and that infered from the trainined
    regressor.

    Parameters
    ----------
    training_df : pandas DataFrame
        DataFrame of training data, a subset of the true distribution.
    classifier_name : str
        Name of the classification algorithm to use.
    N : int
        Sets the (NxN) resolution of points used to query the trained classifier.
    verbose : bool, optional
        Print more useful information.

    Returns
    -------
    pred_class : array
        1D array of predictions from the trained classifier.
    true_class_result : array
        1D array of the true classification for the corresponding points.
    all_regr_abs_frac_diffs_per_class : list
        List of arrays that contain absolute value of fractional differences
        between predicted and analytic regression values.

    Notes
    -----
    This function is written specifically for the original snythetic data set.
    As such, columns and classes are assumed constant and hard coded.
    """
    td = TableData( None, ["input_1", "input_2"], ["class", "output_1"],
                   "class", my_DataFrame=training_df , verbose=False )

    cls_obj = Classifier( td )
    cls_obj.train_everything( [classifier_name], verbose=False )

    regr_obj = Regressor( td )
    regr_obj.train_everything( [regressor_name], verbose=False )

    x_min = np.min(training_df["input_1"]); x_max = np.max(training_df["input_1"])
    y_min = np.min(training_df["input_2"]); y_max = np.max(training_df["input_2"])
    x_vals = np.linspace( x_min, x_max,N); y_vals = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid( x_vals, y_vals )
    stacked_points = np.concatenate( (np.vstack(X.flatten()), np.vstack(Y.flatten())), axis=1  )

    where_nan, where_not_nan = check_grid_points(stacked_points, cls_obj, classifier_name=classifier_name,
                                                verbose=verbose)

    pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions( classifier_name,
                                                                             stacked_points,
                                                                             return_ids=False)

    if len(where_nan) > 0:
        # if there are nans, replace the preds with None so they count as missclassifications
        new_pred_class = np.empty( stacked_points.shape[0], dtype='object' )
        new_pred_class[where_not_nan] = pred_class
        new_pred_class[where_nan] = [None]*len(where_nan)
        pred_class = new_pred_class.copy()

    # Regression
    all_regr_preds_per_cls = []
    all_regr_locs_per_cls = []
    for cls in ["A", "B", "C", "D"]:
        # We look where the predictions are, not the true class result ....
        loc_where_cls = np.where( np.array(pred_class) == cls)[0]
        # Get predictions only for inputs where each class
        regr_preds = regr_obj.get_predictions( [regressor_name], [cls],
                                              ["output_1"], stacked_points[loc_where_cls] )
        regr_key = regr_obj.get_regressor_name_to_key( regressor_name )
        # Save the array of predictions
        all_regr_preds_per_cls.append( regr_preds[regr_key][cls]["output_1"] ) # array
        all_regr_locs_per_cls.append( stacked_points[loc_where_cls] )


    from cris.examples.data.analytic_class_regr import get_output
    true_class_result, true_regr_output = get_output( *stacked_points.T )

    # compare the regression predictions to the true values; calc fractional differences
    all_regr_abs_frac_diffs_per_class = []
    for i, cls in enumerate(["A", "B", "C", "D"]):
        this_cls_true_result, this_cls_true_regr_output = get_output( *all_regr_locs_per_cls[i].T )
        this_cls_diffs = all_regr_preds_per_cls[i] - this_cls_true_regr_output
        all_regr_abs_frac_diffs_per_class.append( abs(this_cls_diffs/this_cls_true_regr_output) )

    where_preds_match_true = np.where( pred_class == true_class_result, 1, 0 )
    accuracy = np.sum(where_preds_match_true)/N**2
    error_rate = 1 - accuracy
    if verbose:
        print( "N training points: {0}, N query points: {1}".format(len(training_df), N*N) )
        print( "accuracy: {}".format(accuracy) )

    return np.array(pred_class), true_class_result, all_regr_abs_frac_diffs_per_class

def check_grid_points( stacked_points, classifier_obj, classifier_name = "linear", verbose = False):
    cls_obj = classifier_obj
    pred_class, max_probs, where_not_nan = \
                cls_obj.get_class_predictions(classifier_name, stacked_points, return_ids=False)

    where_nan = [i for i in range(len(stacked_points)) if i not in where_not_nan]

    if len(where_nan) + len(where_not_nan) != len(stacked_points):
        raise Exception("Where nan and where not nan do not add to total points!")

    if len(where_nan) > 0:
        if verbose:
            print("\tFound {} nans.".format(len(where_nan)), end="\r")
    return where_nan, where_not_nan


def get_confusion_matrix(preds, actual, all_classes, verbose=False):
    """Given a list of predicted values and actual values (output of
    the method get_prediction_diffs), Calculate a confusion matrix.

    Parameters
    ----------
    preds : list
        Predicted values from the classifier.
    actual : list
        True values from the underlying distribution.
    all_classes : list
        A list of all unique classes to be considered.
        Should be either np.unique(actual) or a subset thereof.
    verbose : bool, optional
        Print our the line by line confusion matrix prefixed with the class.
    """
    confusion_matrix = []
    for pred_class_key in all_classes:
        loc = np.where( actual == pred_class_key) # where all true class
        how_many_per_class = np.array([ np.sum(preds[loc]==i) for i in all_classes])
        # [ how many preds matched true class A, how many preds matched true clas B]
        how_many_per_class = how_many_per_class/len( loc[0] ) # normalize
        confusion_matrix.append( how_many_per_class )
        if verbose:
            print(pred_class_key, how_many_per_class)

    return np.array(confusion_matrix)


def calc_performance(dfs_per_iter, cls_name="linear", regr_name="rbf", resolution=400, verbose=False):
    """Given a list of pandas DataFrames, iterate over them and
    calculate the accuracy and confusion matrix for synthetic data sets.

    Parameters
    ----------
    dfs_per_iter : list
        List of pandas DataFrames containing training data
        to train an classifier on and then compare to the
        true background distribution.
    cls_name : str, optional
        Name of classifier to train.
    resolution : int, optional
        Density per axis of the grid used to oversample the true background.
    verbose : bool, optional
        Print some helpful info.

    Returns
    -------
    acc_per_iter : array
        Array conatining overall accuracy of interpolator
        per iteration of training data.
    conf_matrix_per_iter : list
        List of confusion matricies per iteration. Calculated
        using 'get_confusion_matrix'.
    abs_regr_frac_diffs_per_iter : list
        List of fractional differences per class for each DataFrame.
    """

    acc_per_iter = []
    conf_matrix_per_iter = []
    abs_regr_frac_diffs_per_iter = []
    for j, df in enumerate(dfs_per_iter):
        if verbose: print( "\ndf: {0}".format(j) )
        N = resolution
        predictions, true_class_result,  abs_regr_frac_diffs_per_cls = \
                        get_prediction_diffs( df, classifier_name=cls_name,
                                              regressor_name=regr_name,
                                                N=N, verbose=verbose )

        conf_matrix = get_confusion_matrix(predictions, true_class_result, np.unique(true_class_result) )
        conf_matrix_per_iter.append( conf_matrix )

        where_preds_match_truth = np.where( predictions == true_class_result, 1, 0 )
        accuracy = np.sum(where_preds_match_truth)/N**2
        acc_per_iter.append( accuracy )
        abs_regr_frac_diffs_per_iter.append( abs_regr_frac_diffs_per_cls )
    return np.array(acc_per_iter), conf_matrix_per_iter, abs_regr_frac_diffs_per_iter
