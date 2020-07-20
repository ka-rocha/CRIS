# generally useful imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy
from scipy.spatial.distance import cdist

from cris.data import TableData
from cris.classify import Classifier
from cris.regress import Regressor
from cris.sample import Sampler

from cris.examples.synthetic_data.synth_data_2D import get_output_2D
from cris.examples.synthetic_data.synth_data_3D import get_output_3D

# for parsing ini files
from configparser import ConfigParser
from ast import literal_eval

def parse_inifile(path):
    """Parse an ini file to run psy-cris method 'get_new_query_points'.

    Parameters
    ---------
    path : str
        Path to ini file.

    Returns
    -------
    all_kwargs_dict : dict
        Nested dictionary of parsed inifile kwargs.
    """
    all_kwargs_dict = {}
    confparse = ConfigParser()
    confparse.read( path )
    for sect in confparse:
        if sect == "DEFAULT": continue
        section_dict = {}
        for var in confparse[sect]:
            section_dict[var] = literal_eval( confparse[sect][var] )
        all_kwargs_dict[sect + "_kwargs"] = section_dict
    return all_kwargs_dict


def get_new_query_points( N_new_points=1, TableData_kwargs = {},
                                  Classifier_kwargs={}, Regressor_kwargs={},
                                  Sampler_kwargs={}, Proposal_kwargs={}, threshold=1e-5 ):
    """Run the psy-cris algorithm to propose new query points to be labeled.

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
    Proposal_kwargs : dict, optional
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
    if Regressor_kwargs.get("do_regression", False):
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
    where_to_cut = int( len(last_chain_hist) * Proposal_kwargs.get("cut_fraction", 0) )
    last_chain_hist = last_chain_hist[where_to_cut:]

    # propose new points
    classifier_name = Proposal_kwargs.get("pred_classifier_name", "rbf")
    if classifier_name not in Classifier_kwargs.get('classifier_names'):
        raise Exception("Predictions must be with a trained classifier. '{0}' was given".format(classifier_name) )

    kappa = Proposal_kwargs.get("kappa",150)
    proposed_points, final_kappa = sampler_obj.get_proposed_points(last_chain_hist, N_new_points, kappa,
                                                                    **Proposal_kwargs)
    pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions(classifier_name, proposed_points, return_ids=False)
    return proposed_points, pred_class




################################################################################
# Functions below this point are designed specifically for testing cris so they
# assume things like the data set being used and do not have the flexibility
# present throughout the rest of the code.
################################################################################


def do_dynamic_sampling( N_final_points=100, new_points_per_iter=20, verbose=False,
                        threshold=1e-5, N_starting_points=100, jitter=False, dim=2, **all_kwargs  ):
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
    my_data = get_regular_grid_df( N_starting_points, jitter=jitter, dim=dim )
    all_kwargs["TableData_kwargs"]["my_DataFrame"] = my_data
    dfs_per_iters = [my_data]; preds_per_iter = [None]

    N_total = int(N_final_points - N_starting_points)
    if verbose:
        print("Sampling {} total points...".format(N_total))
        print("DIM: {}".format(dim))

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
        if dim == 2:
            random_init_pos = np.random.uniform(low=-2,high=2, size=(2))
        elif dim == 3:
            random_init_pos = np.random.uniform(low=-0.5,high=0.5, size=(3))
        all_kwargs["Sampler_kwargs"]["init_pos"] = random_init_pos

        # Check distances
        if dim == 2:
            old_points = my_data[["input_1", "input_2"]].to_numpy()
        elif dim == 3:
            old_points = my_data[["input_1", "input_2", "input_3"]].to_numpy()
        where_good_bools = check_dist( old_points, new_points, threshold=threshold )
        if np.sum(where_good_bools) != len(new_points):
            print("We are getting rid of {} points below thresh this iter.".format(len(new_points)-np.sum(where_good_bools)) )
        new_points = new_points[where_good_bools]

        # Evaluate query points with anayltic dataset
        if dim == 2:
            output_new_points_df = get_output_2D( *new_points.T )
        elif dim == 3:
            output_new_points_df = get_output_3D( *new_points.T )

        new_data = my_data.append( output_new_points_df, ignore_index=True )
        # Append data to be returned
        dfs_per_iters.append( new_data )
        my_data = new_data.copy()
        # Append data to be returned
        preds_per_iter.append( cls_preds )


        where_to_cut = int(N_starting_points + N_sampled_points)

        if dim == 2:
            plot_proposed_points_2D(where_to_cut, num_loops, random_init_pos, my_data, where_good_bools)
        elif dim == 3:
            plot_proposed_points_3D(where_to_cut, num_loops, random_init_pos, my_data, where_good_bools)

        N_sampled_points += len(new_points)
        num_loops += 1
        if verbose:
            end_of_iter_str = "\tEND ITER {0} in {1:.2f}s".format(num_loops-1, time.time()-start_time)
            print( end_of_iter_str + ", N_sampled_points: {}\n\n".format(N_sampled_points) )

    if verbose:
        print( "\n\nDone. Sampled {0} points.".format(len(my_data)-N_starting_points) )
        print( "Total time: {:.2f}s".format(time.time()-t0)  )

    return dfs_per_iters, preds_per_iter



def plot_proposed_points_2D( where_to_cut, num_loops, random_init_pos, my_data, where_good_bools ):
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


def plot_proposed_points_3D( where_to_cut, num_loops, random_init_pos, my_data, where_good_bools ):
        fig, subs = plt.subplots(1,2, figsize=(8,3.5), dpi=100)
        subs[0].set_title("SAMPLE {0}".format(num_loops))
        subs[0].plot( random_init_pos[0], random_init_pos[1],
                     '+', markeredgewidth=1.5, color="red", label="init_pos")
        subs[0].scatter( my_data["input_1"][0:where_to_cut],
                         my_data["input_2"][0:where_to_cut],
                         alpha=0.5, color = "dodgerblue", label="training")
        subs[0].scatter( my_data["input_1"][where_to_cut:],
                         my_data["input_2"][where_to_cut:],
                         marker = 'x', color = "C2", label="proposed" )
        subs[0].set_xlabel("X - {0} new points this iter".format(np.sum(where_good_bools)))
        subs[0].set_ylabel("Y")


        subs[1].plot( random_init_pos[0], random_init_pos[2],
                     '+', markeredgewidth=1.5, color="red", label="init_pos")
        subs[1].scatter( my_data["input_1"][0:where_to_cut],
                         my_data["input_3"][0:where_to_cut],
                         alpha=0.5, color = "dodgerblue", label="training")
        subs[1].scatter( my_data["input_1"][where_to_cut:],
                         my_data["input_3"][where_to_cut:],
                         marker = 'x', color = "C2", label="proposed" )
        subs[1].set_xlabel("X")
        subs[1].set_ylabel("Z")

        plt.legend(bbox_to_anchor=[1, 0, 0.22,1])
        plt.show()


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


def get_regular_grid_df(N=100, jitter=False, verbose=False, N_ppa=None, dim=2):
    """Given N total points, produce an even grid with
    approximately the same number of evenly spaced points sampled
    from the analytic data set (2D or 3D).

    The number of returned grid points is N only if N is a perfect square.
    Otherwise use N_ppa to define number of points per axis.

    Parameters
    ----------
    N : int
        Total number of points to make into a 2D even grid
    jitter : bool, optional
        Place the center of the grid randomly around (0,0) in the range of
        +/- 1/2 bin width while keeping the span in each axis at 6.
    N_ppa : array, optional
        Numbers of points per axis. If provided, it overrides N.
    dim : int, optional
        Dimensionality of synthetic data set. (2 or 3)
    verbose : bool, optional
        Print some diagnostics.

    Returns
    -------
    extra_points : pandas DataFrame
        DataFrame of true data drawn from the analytic classification
        and regression functions.
    """
    dim = int(dim)
    if dim != 2 and dim != 3:
        raise ValueError( "Dimensionality {} not supported.".format(dim) )

    if N_ppa is None:
        root_of_N = np.round( N**(1.0/dim) )
        x_res = int(root_of_N)
        y_res = int(root_of_N)
        if dim == 3:
            z_res = int(root_of_N)
    else:
        x_res = int(N_ppa[0])
        y_res = int(N_ppa[1])
        if dim == 3:
            z_res = int(N_ppa[2])

    if verbose:
        if dim == 2:
            print("x_res: {0}, y_res: {1}\nx*y:{2}".format(x_res, y_res, x_res*y_res) )
        elif dim == 3:
            print("x_res: {0}, y_res: {1}, z_res{2}\nx*y*z:{3}".format(x_res,y_res,z_res, x_res*y_res*z_res) )

    if jitter:
        if dim == 2:
            bin_widths = 6 / np.array([x_res, y_res]) # span / num bins
        elif dim == 3:
            bin_widths = 2 / np.array([x_res, y_res, z_res]) # span / num bins
        random_center = np.random.uniform( low=(-0.5), high=(0.5), size=(dim) )
        center_point = bin_widths * random_center
    else:
        center_point = np.array([0,0,0])

    if verbose:
        print( "center_point : {}".format(center_point) )

    if dim == 2:
        center_x, center_y = center_point[:2]
        X, Y = np.meshgrid( np.linspace(-3+center_x,3+center_x,x_res),
                            np.linspace(-3+center_y,3+center_y,y_res) )
        return get_output_2D(X,Y)

    elif dim == 3:
        center_x, center_y, center_z = center_point[:3]
        X, Y, Z = np.meshgrid( np.linspace(-1+center_x,1+center_x,x_res),
                               np.linspace(-1+center_y,1+center_y,y_res),
                               np.linspace(-1+center_z,1+center_z,z_res) )
        return get_output_3D(X,Y,Z)



def get_random_grid_df(N, dim=2):
    """Given N total points, produce a randomly sampled grid drawn
    from the analytic data set (2D or 3D).

    Parameters
    ----------
    N : int
        Total number of points to drawn from a 2D random data set
    dim : int
        Dimensionality of synthetic data set. (2 or 3)

    Returns
    -------
    random_df : pandas DataFrame
        DataFrame of true data drawn from the analytic classification
        and regression functions.
    """
    if dim == 2:
        stacked_points = np.random.uniform( low=(-3,-3), high=(3,3), size = (N,2) )
        random_df = get_output_2D( *stacked_points.T )
    elif dim == 3:
        stacked_points = np.random.uniform( low=(-1,-1,-1), high=(1,1,1), size = (N,3) )
        random_df = get_output_3D( *stacked_points.T )
    return random_df

# PERFORMANCE CALCULATIONS

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
        predictions, true_class_result, abs_regr_frac_diffs_per_cls = \
                        get_prediction_diffs( df, classifier_name=cls_name,
                                              regressor_name=regr_name,
                                                N=N, verbose=verbose )

        conf_matrix = get_confusion_matrix(predictions, true_class_result, np.unique(true_class_result) )
        conf_matrix_per_iter.append( conf_matrix )

        where_preds_match_truth = np.where( predictions == true_class_result, 1, 0 )
        dim = len( [val for val in df.columns if "input" in val] )
        accuracy = np.sum(where_preds_match_truth)/N**dim
        acc_per_iter.append( accuracy )
        abs_regr_frac_diffs_per_iter.append( abs_regr_frac_diffs_per_cls )
    return np.array(acc_per_iter), conf_matrix_per_iter, abs_regr_frac_diffs_per_iter

def get_prediction_diffs( training_df, classifier_name = "linear", regressor_name = "linear",
                         N = 400, verbose=False):
    """From a DataFrame of training data, train a classifier and get both
    the predicitons and actual classification in the classification space
    where the analytic function is defined. Also calculate the difference
    between the true regression function and that infered from the trainined
    regressor. Dimensionality is infered from 'training_df'.

    Parameters
    ----------
    training_df : pandas DataFrame
        DataFrame of training data, a subset of the true distribution.
    classifier_name : str
        Name of the classification algorithm to use.
    N : int
        Sets the (N**dim) resolution of points used to query the trained classifier.
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
    """
    start_time = time.time(); og_start_time = time.time()
    input_col_names = [val for val in training_df.columns if "input" in val]
    output_col_names = [val for val in training_df.columns if "output" in val or "class" in val]
    td = TableData( None, input_col_names, output_col_names,
                   "class", my_DataFrame=training_df , verbose=False )

    cls_obj = Classifier( td )
    cls_obj.train_everything( [classifier_name], verbose=False )

    regr_obj = Regressor( td )
    regr_obj.train_everything( [regressor_name], verbose=False )

    timer=kwargs.get("timer", False)
    if timer: print( "PSY-CRIS TRAIN: ",time.time()-start_time )
    start_time = time.time()

    dim = len(input_col_names)
    axes_values = []
    for i, name in enumerate(input_col_names):
        axis_min = np.min(training_df[name])
        axis_max = np.max(training_df[name])
        axis_vals = np.linspace( axis_min, axis_max, N )
        axes_values.append(axis_vals)

    if dim == 2:
        X, Y = np.meshgrid( *axes_values )
        holder = ( X.flatten(), Y.flatten() )
        stacked_points = np.array(holder).T
    elif dim == 3:
        X, Y, Z = np.meshgrid( *axes_values )
        holder = ( X.flatten(), Y.flatten(), Z.flatten() )
        stacked_points = np.array(holder).T

    if timer: print( "VSTACK: ",time.time()-start_time )
    start_time = time.time()

    pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions( classifier_name,
                                                                             stacked_points,
                                                                             return_ids=False)

    all_possible_ilocs = np.array( range(0,len(stacked_points)) )
    # s.difference(t)  new set with elements in s but not in t
    set_where_nan = set( all_possible_ilocs ).difference( where_not_nan )
    where_nan = np.array( list(set_where_nan) )

    if timer: print( "get_class_predictions: ",time.time()-start_time )
    start_time = time.time()

    if len(where_nan) > 0:
        # if there are nans, replace the preds with None so they count as missclassifications
        new_pred_class = np.empty( stacked_points.shape[0], dtype='object' )
        new_pred_class[where_not_nan] = pred_class
        new_pred_class[where_nan] = [None]*len(where_nan)
        pred_class = new_pred_class.copy()

    if timer: print( "where_nan > 0: ",time.time()-start_time )
    start_time = time.time()

    # Regression
    all_regr_preds_per_cls = []
    all_regr_locs_per_cls = []
    all_unique_classes = np.unique( training_df["class"] )
    if verbose:
        print("CLASSES: {}".format(all_unique_classes))
    for cls in all_unique_classes:
        # We look where the predictions are, not the true class result ....
        loc_where_cls = np.where( np.array(pred_class) == cls)[0]
        # Get predictions only for inputs where each class
        regr_preds = regr_obj.get_predictions( [regressor_name], [cls],
                                              ["output_1"], stacked_points[loc_where_cls] )
        regr_key = regr_obj.get_regressor_name_to_key( regressor_name )
        # Save the array of predictions
        all_regr_preds_per_cls.append( regr_preds[regr_key][cls]["output_1"] ) # array
        all_regr_locs_per_cls.append( stacked_points[loc_where_cls] )

    if timer: print( "Regression vals: ",time.time()-start_time )
    start_time = time.time()

    if dim == 2:
        get_output_func = get_raw_output_2D
    elif dim==3:
        get_output_func = get_raw_output_3D

    true_class_result, true_regr_output = get_output_func( *stacked_points.T )

    # compare the regression predictions to the true values; calc fractional differences
    all_regr_abs_frac_diffs_per_class = []
    for i, cls in enumerate(all_unique_classes):
        this_cls_true_result, this_cls_true_regr_output = get_output_func( *all_regr_locs_per_cls[i].T )
        this_cls_diffs = all_regr_preds_per_cls[i] - this_cls_true_regr_output
        all_regr_abs_frac_diffs_per_class.append( abs(this_cls_diffs/this_cls_true_regr_output) )

    if timer: print( "regr diffs: ",time.time()-start_time )
    start_time = time.time()


    where_preds_match_true = np.where( pred_class == true_class_result, 1, 0 )
    accuracy = np.sum(where_preds_match_true)/N**(dim)
    error_rate = 1 - accuracy
    if verbose:
        print( "N training points: {0}, N query points: {1}".format(len(training_df), N**(dim)) )
        print( "accuracy: {}".format(accuracy) )

    print( "Final stuff END: ", time.time()-start_time )
    print("TOTAL", time.time() - og_start_time)

    return np.array(pred_class), true_class_result, all_regr_abs_frac_diffs_per_class


def get_confusion_matrix(preds, actual, all_classes, verbose=False):
    """Given a list of predicted values and actual values. Calculate a confusion matrix.

    Parameters
    ----------
    preds : list
        Predicted values from the classifier.
    actual : list
        True values from the underlying distribution.
    all_classes : list
        A list of all unique classes.
        Should be either np.unique(actual) or a subset thereof.
    verbose : bool, optional
        Print our the line by line confusion matrix prefixed with the class.

    Returns
    -------
    confusion_matrix : ndarray
        Rows and columns of confusion matrix in order and number given in 'all_classes'.
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
