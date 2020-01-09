import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import OrderedDict

# -------- regressors --------
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf

import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# -----------------------------


LinearNDInterpolator_names = ["linear", "linearndinterpolator","linear nd interpolator"]
RBF_names = ["rbf", "radialbasisfunction", "radial basis function"]
GaussianProcessRegressor_names = ["gp", "gpr", "gaussianprocessregressor"]

# wow
# https://stackoverflow.com/questions/27118687/updating-nested-dictionaries-when-data-has-existing-key/27118776
import collections
def makehash():
    return collections.defaultdict(makehash)


class Regressor():
    """
    Regressor
    =========

    Perform regression and interpolation with a variety of different
    regression algorithms.

    Parameters
    ----------
    TableData_object : instance of <class, TableData>
        An instance of the TableData class.


    ADD DOCS
    """

    def __init__(self, table_data):
        self._TableData_ = TableData_object

        holder = self._TableData_.get_regr_data(what_data='full')
        self.input_dict = holder[0] #_regr_inputs_
        self.output_dict = holder[1] #_regr_outputs_
        self.regr_dfs_per_class = holder[2] #_regr_dfs_per_class_

        self._regressors_ = makehash()
        self._cv_regressors_ = makehash()

        self._log_history_ = makehash()
        self._cv_log_history = makehash()

        self.__train_cross_val = False # TODO: need to still update everything
        self.__all_diffs_holder = makehash()
        self.__all_Pchange_holder = makehash()

    def train_everything(self, regressor_names, verbose=False):
        """Train all classes and columns with the specified list of regressor names."""
        for regr_name in regressor_names:
            if verbose:
                print("Regressor: {0}".format(regr_name))
            class_keys=list(self.regr_dfs_per_class.keys())
            for class_name in class_keys:
                self.train(regr_name, [class_name], None, verbose=verbose)
        if verbose:
            print("\nDone Regrssor train_everything.")
        return None


    def train(self, regressor_name, class_keys, col_keys, di = None, verbose = False, train_cross_val = False ):
        """Train a regression algorithm.

        Can choose mutliple classes / columns at once as long as
        they have the same columns.

        if col_keys is None then it trains on all columns in one class.
        """
        regressor_key = self.get_regressor_name_to_key(regressor_name)

        if col_keys is None:
            first_class_data = self.regr_dfs_per_class[class_keys[0]]
            if isinstance(first_class_data, pd.DataFrame):
                col_keys = np.array( first_class_data.keys() )
                if verbose:
                    print("\t Training on all {0} columns in '{1}'...".format(len(col_keys),class_keys[0]) )
            else:
                if verbose:
                    print("No regression data for {0}.".format(class_keys[0]))
                return

        if   regressor_key == "LinearNDInterpolator":
            regr_holder = self.fit_linear_ND_interpolator( class_keys, col_keys, data_interval = di, verbose = verbose )
        elif regressor_key == "RBF":
            regr_holder = self.fit_rbf_interpolator( class_keys, col_keys, data_interval = di, verbose = verbose )
        elif regressor_key == "GaussianProcessRegressor":
            regr_holder = self.fit_gaussian_process_regressor( class_keys, col_keys, data_interval = di, verbose = verbose )
        else:
            print("No trainers with name {0}".format(regressor_name))
            return

        for class_key, class_dict in regr_holder.items():
            for col_key, interpolated_obj in class_dict.items():
                if verbose:
                    print('\tdict loc: {0}, {1}, {2},'.format(regressor_key, class_key, col_key) )
                if train_cross_val:
                    self._cv_regressors_[regressor_key][class_key][col_key] = interpolated_obj
                else:
                    self._regressors_[regressor_key][class_key][col_key] = interpolated_obj

        if verbose:
            print("\tEXIT TRAIN\n")
        return None


    def fit_linear_ND_interpolator(self, class_keys, col_keys, data_interval = None, verbose = False):
        """See GRP"""
        if verbose:
            print("--- Fit LinearNDInterpolator ---")

        start_time = time.time()
        regressor_holder = OrderedDict()

        for class_key in class_keys:
            this_class_dict = OrderedDict() # will hold columns

            # extract the output data associated with class_key
            which_class_data = self.regr_dfs_per_class[class_key]

            for col_key in col_keys:

                if  data_interval is None:
                    training_x = self.input_dict[class_key].to_numpy(float)
                    training_y = which_class_data[col_key].to_numpy(float)
                else:
                    di = np.array(data_interval)
                    training_x = self.input_dict[class_key].to_numpy(float)[di]
                    training_y = which_class_data[col_key].to_numpy(float)[di]

                if verbose:
                    print("%s: %s - %.0f training points"%(class_key, col_key, len(training_x)) )

                line = LinearNDInterpolator( training_x, training_y )

                this_class_dict[col_key] = line
            regressor_holder[class_key] = this_class_dict

        if verbose:
            print("--- Done in {0:.2f} seconds. ---".format( time.time() - start_time) )

        return regressor_holder


    def fit_rbf_interpolator(self, class_keys, col_keys, data_interval = None, verbose = False):
        """See GPR"""
        if verbose:
            print("--- Fit RBF ---")

        start_time = time.time()
        regressor_holder = OrderedDict()

        for class_key in class_keys:
            this_class_dict = OrderedDict() # will hold columns

            # extract the output data associated with class_key
            which_class_data = self.regr_dfs_per_class[class_key]

            for col_key in col_keys:

                if  data_interval is None:
                    training_x = self.input_dict[class_key].to_numpy(float)
                    training_y = which_class_data[col_key].to_numpy(float)
                else:
                    di = np.array(data_interval)
                    training_x = self.input_dict[class_key].to_numpy(float)[di]
                    training_y = which_class_data[col_key].to_numpy(float)[di]

                argList = []
                for col in range( len(training_x[0]) ):
                    argList.append( training_x.T[col] )
                argList.append( training_y )

                if verbose:
                    print("%s: %s - %.0f training points"%(class_key, col_key, len(training_x)) )

                line = Rbf( *argList )

                this_class_dict[col_key] = line
            regressor_holder[class_key] = this_class_dict

        if verbose:
            print("--- Done in %.2f seconds. ---"%( time.time() - start_time) )

        return regressor_holder


    def fit_gaussian_process_regressor(self, class_keys, col_keys, data_interval = None, verbose = False):
        """ Ok so this is gonna create a dict sorted by class and each
        element is then another dict with the column names mapping to
        gp regressor objects """
        if verbose:
            print("--- Fit GaussianProcessRegressor ---")

        start_time = time.time()

        n_restarts = 3

        regressor_holder = OrderedDict()

        for class_key in class_keys:
            this_class_dict = OrderedDict() # will hold columns

            # extract the output data associated with class_key
            which_class_data = self.regr_dfs_per_class[class_key]

            for col_key in col_keys:

                if  data_interval is None:
                    training_x = self.input_dict[class_key].to_numpy(float)
                    training_y = which_class_data[col_key].to_numpy(float)
                else:
                    di = np.array(data_interval)
                    training_x = self.input_dict[class_key].to_numpy(float)[di]
                    training_y = which_class_data[col_key].to_numpy(float)[di]

                if verbose:
                    print("%s: %s - %.0f training points"%( class_key, col_key, len(training_x)) )

                num_dim = len(training_x[0])
                starting_loc = [1 for i in range(num_dim)]
                axis_range = axis_ranges = [ (1e-3,1e3) for i in range(num_dim) ]
                #kernel = C( 1e3, (1e2, 5e4) ) * RBF( [ 10, 500, 300.], [(1e0,1e3), (1e0,1e3), (1e-1, 5e3)] )
                kernel = gp.kernels.RBF( starting_loc, axis_ranges )
                gpr = gp.GaussianProcessRegressor( kernel=kernel, n_restarts_optimizer = n_restarts )

                #print("PRE-fit params:\n", gpr.kernel.get_params() ) # helpful for kernel things
                gpr.fit( training_x, training_y)
                if verbose:
                    print("POST-fit params:\n", gpr.kernel_.get_params() )

                this_class_dict[col_key] = gpr
            regressor_holder[class_key] = this_class_dict

        if verbose:
            print("--- Done in %.2f seconds. ---"%( time.time() - start_time) )

        return regressor_holder


    def get_predictions(self, regressor_names, class_keys, col_keys, test_input, return_std = False, cross_val = False):
        """Dicts the same format as the regression objects but instead
        the are filled with the regressed output!"""

        predictions = OrderedDict()

        for regr_name in regressor_names:
            regr_key = self.get_regressor_name_to_key(regr_name)
            this_class_dict = OrderedDict()
            for class_key in class_keys:
                these_cols_dict = OrderedDict()

                for col_key in col_keys:
                    pred_vals = self._predict( regr_key, class_key, col_key, test_input, \
                                            return_std = return_std, cross_val = cross_val )
                    these_cols_dict[col_key] = pred_vals

                this_class_dict[class_key] = these_cols_dict
            predictions[regr_key] = this_class_dict

        return predictions


    def _predict(self, regressor_name, class_key, col_key, test_input, return_std = False, cross_val = False):

        sigma = None # default

        if not bool(self._regressors_) and not bool( self._cv_regressors_ ): # if empty
            raise Exception("\n\nNo trained interpolators exist.")

        regressor_key = self.get_regressor_name_to_key(regressor_name)

        if cross_val:
            interpolators = self._cv_regressors_[regressor_key]
        else:
            interpolators = self._regressors_[regressor_key]

        interp = interpolators[class_key][col_key]

        if regressor_key == "RBF":
            argList = []
            for col in range( len(test_input[0]) ):
                argList.append( test_input.T[col] )
            pred = interp( *argList )
        elif regressor_key == "GaussianProcessRegressor":
            if return_std:
                pred, sigma = interp.predict(test_input, return_std = True)
            else:
                pred = interp.predict(test_input)
        elif regressor_key == "LinearNDInterpolator":
            pred = interp( test_input )
        else:
            print("Name not recognized: {0}".format(regressor_name))

        if return_std:
            return np.array(pred), np.array(sigma)
        else:
            return np.array(pred)


    def get_regressor_name_to_key(self, name):
        """Return the standard key (str) of a classifier."""
        if   name.lower() in LinearNDInterpolator_names:
            key = "LinearNDInterpolator"
        elif name.lower() in RBF_names:
            key = "RBF"
        elif name.lower() in GaussianProcessRegressor_names:
            key = "GaussianProcessRegressor"
        else:
            print("No regressor with name '%s'."%name)
            return None
        return key


    # def get_structure(self, args):
    #     return None


    def make_cross_val_data(self, class_key, col_key, alpha):
        """Randomly sample the data set and seperate training and test data.

        Parameters
        ----------
        class_key :
        col_key :
        alpha : float
            Fraction of data set to use for training. (0.05 = 5% of data set)

        Returns
        -------
        sorted_rnd_int_vals : array
            Indicies which will be used as training points.
        """

        num_points = int( len( self.input_dict[class_key] )*alpha )
        rnd_input_train = []; rnd_outout_train = []; rnd_int_vals = []
        rnd_int_set = set()

        #print("Num points", num_points)
        if alpha > 1 or alpha <= 0:
            raise ValueError("Alpha must be in the range (0,1].")

        ct = 0
        while len(rnd_int_vals) < num_points and ct < 1e7:
            rnd_int = ( int( np.random.random()*len(self.input_dict[class_key]) ) )

            if rnd_int not in rnd_int_set:
                rnd_int_vals.append( rnd_int )
                rnd_int_set.add( rnd_int )
            ct += 1

        train_rnd_int_vals = np.array( sorted(rnd_int_vals) )

        # Random training data
        self.cross_val_train_input_data = ( self.input_dict[class_key].to_numpy(float) )[train_rnd_int_vals,:]
        self.cross_val_train_class_data = ( self.regr_dfs_per_class[class_key][col_key].to_numpy(float) )[train_rnd_int_vals]

        test_int_vals = []
        for i in range(len(self.input_dict[class_key])):
            if i in train_rnd_int_vals:
                pass
            else:
                test_int_vals.append( i )

        # The remainder which will be used to test fits
        self.cross_val_test_input_data = (self.input_dict[class_key].to_numpy(float))[test_int_vals,:]
        self.cross_val_test_output_data = (self.regr_dfs_per_class[class_key][col_key].to_numpy(float))[test_int_vals]

        return train_rnd_int_vals, test_int_vals


    def cross_validate(self, regressor_name, class_key, col_key, alpha, verbose = False):
        """This is not really cross validation - taking differences"""
        self.__train_cross_val = True

        train_data_indicies, test_data_indicies = self.make_cross_val_data( class_key, col_key, alpha )

        if verbose:
            print("alpha: %f, num_training_points %.0f"%(alpha, len(train_data_indicies)) )

        regressor_key = self.get_regressor_name_to_key(regressor_name)

        # Train classifier
        start_time = time.time()
        if regressor_key == "LinearNDInterpolator":
            # if linear - train rbf to use if linear predicts nan
            self.train( regressor_key, [class_key], [col_key], \
                        di = train_data_indicies, train_cross_val = True,
                        verbose = verbose  )
            self.train( "RBF", [class_key], [col_key],\
                        di = train_data_indicies, train_cross_val = True,
                        verbose = verbose)
        else:
            self.train( regressor_key, [class_key], [col_key], \
                        di = train_data_indicies, train_cross_val = True,
                        verbose = verbose  )
        time_to_train = time.time() - start_time

        # Make Predictions
        if regressor_key == "LinearNDInterpolator":
            predicted_values_linear = self._predict( regressor_key, class_key, col_key, \
                                        self.cross_val_test_input_data, cross_val = True )
            predicted_values_rbf = self._predict( "RBF", class_key, col_key, \
                                     self.cross_val_test_input_data, cross_val = True )
            where_nan = np.where( np.isnan(predicted_values_linear) )[0]
            if len(where_nan) > 0:
                print("{0}: {1} nan points out of {2}. Used rbf instead.".format(regressor_key, len(where_nan), len(predicted_values_linear)) )
                predicted_values_linear[where_nan] = predicted_values_rbf[where_nan]
            predicted_values = predicted_values_linear
        else:
            predicted_values = self._predict( regressor_key, class_key, col_key, \
                                    self.cross_val_test_input_data, cross_val = True )

        # Calculate the difference
        diffs = predicted_values - self.cross_val_test_output_data

        where_zero = np.where( self.cross_val_test_output_data == 0 )[0] # 1d array
        where_not_zero = np.where( self.cross_val_test_output_data != 0 )[0] # 1d array

        if len(where_zero) > 0:
            percent_diffs = (diffs[where_not_zero] / self.cross_val_test_output_data[where_not_zero]) * 100
            print("{0} output with value zero. Omitting for percent change calculation.".format(len(where_zero)))
        else:
            percent_diffs = (diffs / self.cross_val_test_output_data) * 100

        self.__train_cross_val = False
        return percent_diffs, diffs

    #   Get it completley working!!!!!
    #   Nearest Neighbors differences
    #       extra_column = (%diff_1+%diff_2+%diff_3 +...)/N
    #   do it in data.py
    #   interpolate using regressors
    #   update target distribution (add them)
    #   - fix Milena's code (day max)
    #   !!! :D

    def get_frac_diffs(self, regressor_name, class_key, args, which_cols=None, verbose=False ):
        """Get desired fractional differences calculated in DataFrame object.

        # TODO: finish this method

        Parameters
        ----------
        regressor_name : str
            Name of regression method to use for interpolating between differences
            data.
        class_key : str
            Key corresponding to a unique class from the data set.
        which_cols : list, optional
            List of substrings which pick out columns used for differences.
        verbose : bool, optional

        Returns
        -------

        """
        regressor_key = self.get_regressor_name_to_key(regressor_name)

        if isinstance(self.regr_dfs_per_class[class_key], pd.DataFrame):
            all_apc_keys = [i for i in self.regr_dfs_per_class[class_key].keys() if "APC" in i]
            all_prefixes = np.array([i.split("_", 2) for i in all_apc_keys])
            unique_prefix = np.unique(all_prefixes.T[0])

            if which_cols is None:
                good_col_keys = all_apc_keys
            else:
                for sub_str in which_cols:
                    good_col_keys = [i for i in all_apc_keys if sub_str in i]

            correct_shape_args = np.array([args])
            predictions = self.get_predictions( [regressor_key], [class_key], good_col_keys, correct_shape_args  )
            return None

        else:
            # nan value, return zero
            return 0
        all_cols = self.regr_dfs_per_class[regressor_key]


    def mult_diffs(self, regressor_name, class_key, col_keys, alpha, cutoff, verbose = False):
        #col_keys = self.regr_dfs_per_class[class_key].keys()
        if verbose:
            print("MULT DIFFS:",regressor_name, col_keys)

        p_diffs_holder = []
        for col_key in col_keys:
            p_diffs, diffs = self.cross_validate( regressor_name, class_key, col_key,\
                                                  alpha, verbose = verbose)
            where_not_nan = np.where( np.invert(np.isnan(p_diffs)) )[0]

            p_diffs_holder.append( p_diffs[where_not_nan] )

        attr_holder = []
        for p_diff in p_diffs_holder:
            holder = []

            outside_cutoff = abs(p_diff) >= cutoff*100
            num_outside = np.sum(outside_cutoff)

            holder.append( num_outside/len(p_diff) * 100 ) # percent outside
            holder.append( np.mean(p_diff) ) # mean
            holder.append( np.std(p_diff) ) # standard deviation

            attr_holder.append( holder )

        return np.array(p_diffs_holder), np.array(attr_holder)


    def plot_regr_data(self, class_name):
        """Plot all regression data from the chosen class.

        Parameters
        ----------
        class_name : str
            Specify what class data will plotted.
        """

        data_out = self.regr_dfs_per_class[class_name]
        data_in = self.input_dict[class_name]

        if isinstance(data_out, pd.DataFrame):
            pass
        else:
            print("Output for class '{0}': {1} \nNo valid data to plot.".format(class_name, str(data_out)) )
            return

        key_in  = np.array(data_in.columns)
        key_out = np.array(data_out.columns)

        # note they are still data frames until this point
        num_x_axis = len( data_in.keys() )
        num_y_axis = len( data_out.keys() )

        # inches per subplot - these ratios can be changed
        fig_x_ratio = (4+1/3)
        fig_y_ratio = (3+1/3)

        fig, subs = plt.subplots( nrows=num_y_axis, ncols=num_x_axis, \
                                  dpi=100, figsize=( fig_x_ratio*num_x_axis, fig_y_ratio*num_y_axis ) )

        # so that the indexing below works
        if num_y_axis == 1:
            subs = np.array([subs])

        print("Plotting all regression data from class '{0}'.".format(class_name))

        for i in range( num_x_axis ):
            for k in range( num_y_axis ):
                data_x = np.array( data_in[key_in[i]] ).astype(float)
                data_y = np.array( data_out[key_out[k]] ).astype(float)

                subs[k,i].plot( data_x, data_y, '.' )
                subs[k,i].set_xlabel( key_in[i] )
                subs[k,i].set_ylabel( key_out[k] )
        fig.tight_layout()
        plt.show()
        return None


    def get_rnd_test_inputs(self, class_name, N, other_rng=dict(), verbose=False):
        """Produce randomly sampled 'test' inputs inside domain of input_data.
        Input data is seperated by class.

        Parameters
        ----------
        class_name : str
            Class name to specify which input data you want to look at.
        N : int
            Number of test inputs to return.
        other_rng: dict, optional
            Change the range of random sampling in desired axis. By default,
            the sampling is done in the range of the training data.
            The axis is specified with an integer key and the value
            is a list specifying the range. {1:[min, max]}
        verbose: bool, optional
            Print diagnostic information. (default False)

        Returns
        -------
        rnd_test_points : ndarray
            Test points randomly sampled in the range of the training data
            in each axis unless otherwise specified in 'other_rng'.
            Has the same shape as input data from TableData.
        """

        num_axis = len( self.input_dict[class_name].values[0])

        # find max and min in each axis
        a_max = []; a_min = []
        for i in range(num_axis):
            a_max.append( max(self.input_dict[class_name].values.T[i]) )
            a_min.append( min(self.input_dict[class_name].values.T[i]) )
        # sample N points between max & min in each axis
        axis_rnd_points = []
        for i in range(num_axis):
            if i in other_rng:
                b_min, b_max = other_rng[i]
            else:
                b_min, b_max = a_min[i], a_max[i]
            if verbose:
                print("{0} - min: {1}, max: {2}".format(i, b_min, b_max) )

            r = np.random.uniform( low = b_min, high = b_max, size = N )

            # this reshape is necessary to concatenate
            axis_rnd_points.append( r[:,np.newaxis] )

        # now put the random points back together with same shape as input_data
        rnd_test_points = np.concatenate( axis_rnd_points, axis = 1  )
        return rnd_test_points
