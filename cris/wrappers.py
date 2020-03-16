from . import data
from . import classify
from . import regress
from . import sample

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

    table_obj = data.TableData( files, input_cols, output_cols, class_col,
                                ignore_lines = kwargs.get("ignore_lines", 0 ),
                                verbose = kwargs.get("verbose", False),
                                omit_vals = kwargs.get("omit_vals", ['error', 'No_Run']), \
                                n_neighbors=kwargs.get("n_neighbors", [4]))

    cls_obj = classify.Classifier(table_obj)
    cls_obj.train_everything(['rbf', 'linear'], verbose=False)

    regr_obj = regress.Regressor(table_obj)
    regr_obj.train_everything(["rbf", 'linear'], verbose=False)

    sampler_obj = sample.Sampler( classifier = cls_obj, regressor = regr_obj )

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
    files = TableData_kwargs.pop("file_paths")
    input_cols = TableData_kwargs.pop("input_cols")
    output_cols = TableData_kwargs.pop("output_cols")
    class_col = TableData_kwargs.pop("class_col")

    table_obj = TableData( files, input_cols, output_cols, class_col,
                           **TableData_kwargs)

    # Classifier
    cls_obj = Classifier(table_obj)
    for name in Classifier_kwargs.pop("classifier_names", ["rbf"]):
        cls_obj.train( name, **Classifier_kwargs)

    # Regressor
    regr_obj = Regressor(table_obj)
    regr_obj.train_everything( Regressor_kwargs.pop("regressor_names", ["rbf"]), **Regressor_kwargs)

    # Sampler
    if Sampler_kwargs.pop("init_with_class_and_regression", True):
        sampler_obj = Sampler( classifier = cls_obj, regressor = regr_obj )
    else:
        sampler_obj = Sampler( classifier = cls_obj, regressor = None )

    T_max = Sampler_kwargs.pop("T_max", 50)
    N_tot = Sampler_kwargs.pop("N_tot", 500)
    init_pos = Sampler_kwargs.pop("init_pos", [0,0])
    target_dist_name = Sampler_kwargs.pop("target_dist_name", "TD_classification")
    if target_dist_name == "TD_classification":
        target_dist = sampler_obj.TD_classification
    elif target_dist_name == "TD_classification_regression":
        target_dist = sampler_obj.TD_classification_regression
    elif target_dist_name == "TD_2d_analytic":
        target_dist = sampler_obj.TD_2d_analytic
    else:
        raise ValueError("Use a supported target distribution.")

    interpolation_name = Sampler_kwargs.pop("interpolation_name", "rbf")
    if target_dist_name == "TD_classification_regression":
        if isinstance(interpolation_name, list):
            assert len(interpolation_name)==2
        else:
            raise ValueError("TD_classification_regression requires two strings in a list for 'interpolation_name'.")

    # Sampler.run_PTMCMC
    chain_step_history, T_list = sampler_obj.run_PTMCMC( T_max, N_tot, init_pos, target_dist,
                                                        interpolation_name, **Sampler_kwargs )
    last_chain_hist = chain_step_history[len(T_list)-1]

    # propose new points
    if N_new_points == 1:
        proposed_point = last_chain_hist[-1]
        cls_name = Propose_kwargs.pop("pred_classifier_name", "rbf")
        pred_class, max_probs, where_not_nan = cls_obj.get_class_predictions(cls_name, proposed_point)
        return proposed_point, pred_class
    else:
        kappa = Propose_kwargs.pop("kappa",100)

        proposed_points, kappa = sampler_obj.get_proposed_points(last_chain_hist, N_new_points, kappa,
                                                                 **Propose_kwargs)

        cls_names = Propose_kwargs.pop("pred_classifier_name", "rbf")
        pred_classes, max_probs, where_not_nan = cls_obj.get_class_predictions(cls_names, proposed_points)
        return proposed_points, pred_classes
