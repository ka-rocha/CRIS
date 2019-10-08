import data
import classify
import regress
import sample


def populate_parameter_space( file_path, N_new_points=1 ):
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

    input_cols = ['qratio(M_2i/M_1i)', 'log10(P_i)(days)']
    output_cols = ['result', 'log_L_1', 'log_T_1', 'M_1f(Msun)', \
                   'M_2f(Msun)', 'Porb_f(d)', 'tmerge(Gyr)', \
                   'He_core_1(Msun)', 'C_core_1(Msun)']
    class_col = 'result' # single colum where we want to id different classes

    table_obj = data.TableData( files, input_cols, output_cols, class_col,
                                ignore_lines = 1, verbose = False,
                                omit_vals = ['error', 'No_Run'], n_neighbors=[4] )

    cls_obj = classify.Classifier(table_obj)
    cls_obj.train_everything(['rbf', 'linear'], verbose=False)

    regr_obj = regress.Regressor(table_obj)
    regr_obj.train_everything(["rbf", 'linear'], verbose=False)

    sampler_obj = sample.Sampler( classifier = cls_obj, regressor = regr_obj )

    init_pos = [ [0.5, 0.5] ]
    alpha = [0.05,0.05]

    chain_holder, T_list = sampler_obj.run_PTMCMC( 50, 3000, init_pos[0],
                                    sampler_obj.classifier_target_dist,
                                    'rbf', N_draws_per_swap=5,
                                    c_spacing = 1.5, alpha = alpha,
                                    verbose=True )

    if N_new_points == 1:
        return np.array([chain_holder[len(T_list)-1][-1]])
    else:
        prop_points, Kappa = sampler_obj.get_proposed_points(chain_holder[8], N_new_points, 10, \
                                                                verbose = True)
        return prop_points
