import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import pandas as pd

from scipy.interpolate import LinearNDInterpolator
import time

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Regressor():

    def __init__(self, table_data):
        self.input_dict = table_data.get_regr_input_data() # dict
        self.output_dict = table_data.get_regr_output_data() # dict
        self.class_names =  table_data.get_class_names()

        self.regr_dfs_per_class = table_data.get_regr_sorted_output_data() #dict

        self._regressors_ = dict()

    def train_GP_regressor(self, class_keys, col_keys, n_restarts = 1):
        """ Ok so this is gonna create a dict sorted by class and each
        element is then another dict with the column names mapping to
        gp regressor objects """

        start_time = time.time()

        for class_key in class_keys:
            this_class_dict = dict() # will hold columns
            for col_key in col_keys:
                # extract the output data associated with class_key
                which_class_data = self.regr_dfs_per_class[class_key]

                training_x = np.array( self.input_dict[class_key] )
                training_y = np.array( which_class_data[col_key] )

                print("%s: %s - %.0f data points"%(class_key, col_key, len(training_x)) )

                kernel = C( 1e3, (1e2, 5e4) ) * RBF( [ 10, 500, 300.], [(1e0,1e3), (1e0,1e3), (1e-1, 5e3)] )
                gp = GaussianProcessRegressor( kernel=kernel, n_restarts_optimizer = n_restarts )

                #print("PRE-fit params:\n", gp.kernel.get_params() )
                gp.fit( training_x, training_y)
                #print("POST-fit params:\n", gp.kernel_.get_params() )

                this_class_dict[col_key] = gp

            self._regressors_[class_key] = this_class_dict

        print("Done in %f seconds.\n"%( time.time() - start_time) )





    def plot_data(self, class_name):
        """Plot all regression data from the chosen class."""

        data_out = self.regr_dfs_per_class[class_name]
        data_in = self.input_dict[class_name]

        if isinstance(data_out, pd.DataFrame):
            pass
        else:
            print("Output for class '%s': %s \nNo valid data to plot."%(class_name, str(data_out)))
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

        print("Plotting all regression data from class '%s'."%(class_name))

        for i in range( num_x_axis ):
            for k in range( num_y_axis ):
                data_x = np.array( data_in[key_in[i]] ).astype(float)
                data_y = np.array( data_out[key_out[k]] ).astype(float)

                subs[k,i].plot( data_x, data_y, '.' )
                subs[k,i].set_xlabel( key_in[i] )
                subs[k,i].set_ylabel( key_out[k] )

        fig.tight_layout()
        plt.show()
