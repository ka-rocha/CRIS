from cris.data import TableData
from cris.classify import Classifier
from cris.regress import Regressor
from cris.sample import Sampler

# generally useful imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def populate_parameter_space( file_path, N_new_points=1, **kwargs ):
    """For a given set of training data, propose new simulations
    in parameter space based off the CRIS sampling algorithm.

    A PTMCMC with a target distribution: (1-P(class))+max(frac_diff_in_regr)

    Parameters
    ----------
    file_path : str, list
        String or list of strings with file paths to training
        data for the TableData class.
    N_new_points : int, optional
        Number of new points to propose. If 1 (default) then
        the last position in the PTMCMC is given. If more than
        one, proposed points are based off of density logic.
        (see method 'get_proposed_points' in Sampler class)

    Returns
    -------
    prop_points : ndarray
        Array containing the proposed simulation points.
        Each element is a ND vector with dimensionality given
        by the input data.
    """
    if isinstance(file_path, list):
        files = file_path
    else:
        files = [file_path]

    def_input_cols = ['qratio(M_2i/M_1i)', 'log10(P_i)(days)']
    def_output_cols = ['result', 'log_L_1', 'log_T_1', 'M_1f(Msun)', \
                   'M_2f(Msun)', 'Porb_f(d)', 'tmerge(Gyr)', \
                   'He_core_1(Msun)', 'C_core_1(Msun)']
    def_class_col = 'result'

    input_cols = kwargs.get("input_cols", def_input_cols)
    output_cols = kwargs.get("output_cols", def_output_cols)
    class_col = kwargs.get("class_col", def_class_col) # single colum where we want to id different classes

    table_obj = TableData( files, input_cols, output_cols, class_col,
                                ignore_lines = kwargs.get("ignore_lines", 0 ),
                                verbose = kwargs.get("verbose", False),
                                omit_vals = kwargs.get("omit_vals", ['error', 'No_Run']), \
                                n_neighbors=kwargs.get("n_neighbors", [4]))

    cls_obj = Classifier(table_obj)
    cls_obj.train_everything(['rbf', 'linear'], verbose=False)

    regr_obj = Regressor(table_obj)
    regr_obj.train_everything(["rbf", 'linear'], verbose=False)

    sampler_obj = Sampler( classifier = cls_obj, regressor = regr_obj )

    init_pos = kwargs.get("init_pos", np.array([[0.5, 0.5]]) )
    alpha = kwargs.get("alpha", init_pos[0]/25 )

    chain_holder, T_list = sampler_obj.run_PTMCMC( 50, kwargs.get("PTMCMC_iters", 3000),
                                    init_pos[0],
                                    sampler_obj.classifier_target_dist,
                                    'rbf', N_draws_per_swap=5,
                                    c_spacing = 1.5, alpha = alpha,
                                    verbose=True )

    lowest_T_chain = chain_holder[len(T_list)-1]
    if N_new_points == 1:
        prop_points = np.array([lowest_T_chain[-1]])
    else:
        prop_points, Kappa = sampler_obj.get_proposed_points( lowest_T_chain, N_new_points, 10, \
                                                                verbose = True)

    pred_classes, probs, where_not_nan = cls_obj.get_class_predictions( 'rbf', prop_points, return_ids=False)

    return prop_points, pred_classes









# I think this function above me should be named get proposed points
# And then add another function called populate parameter space that iterates over proposed points
# unitl some convergence on the fake data set or something


def new_populate_parameter_space( N_new_points=1, TableData_kwargs = {},
                                  Classifier_kwargs={}, Regressor_kwargs={},
                                  Sampler_kwargs={}, Propose_kwargs={} ):
    """Testing"""

    # TableData
    table_obj = TableData(**TableData_kwargs)

    # Classifier
    cls_obj = Classifier(table_obj)
    cls_obj.train_everything(**Classifier_kwargs)

    # Regressor
    regr_obj = Regressor(table_obj)
    regr_obj.train_everything(**Regressor_kwargs)

    # Sampler
    sampler_obj = Sampler( classifier = cls_obj, regressor = regr_obj )

    target_dist_name = Sampler_kwargs.get("target_dist", "TD_classification")
    exec( "target_dist_obj =  sampler_obj.{0:s}".format(target_dist_name) )
    exec( "Sampler_kwargs['target_dist'] = target_dist_obj" )
    # Sampler.run_PTMCMC
    chain_step_history, T_list = sampler_obj.run_PTMCMC(**Sampler_kwargs)
    last_chain_hist = chain_step_history[len(T_list)-1]

    # propose new points
    if N_new_points == 1:
        proposed_point = last_chain_hist[-1]
        cls_name = Propose_kwargs.get("pred_classifier_name", "rbf")
        pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions(cls_name, proposed_point)
        return proposed_point, pred_class
    else:
        kappa = Propose_kwargs.get("kappa",150)
        proposed_points, kappa = sampler_obj.get_proposed_points(last_chain_hist, N_new_points, kappa,
                                                                 **Propose_kwargs)

        cls_names = Propose_kwargs.pop("pred_classifier_name", "rbf")
        pred_classes, max_probs, where_not_nan = cls_obj.get_class_predictions(cls_names, proposed_points)
        return proposed_points, pred_classes
