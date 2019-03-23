import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time

# --- classifiers ---
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf
import sklearn.gaussian_process as gp
# -------------------

class Classifier():

    def __init__( self, table_data ):

        self.table_data = table_data

        self.class_names = table_data.get_class_names()
        self.class_ids = table_data.get_class_ids()
        self.classes_to_ids = table_data.get_classes_to_ids()
        self.class_data = table_data.get_class_data()
        self.input_data = table_data.get_input_data()

        self._interpolators_ = dict()


    def linear_ND_interpolator(self, data_interval = None):
        """data_interval = None or int"""

        start_time = time.time()
        line_holder = dict()

        for i, cl in enumerate(self.class_data):
            iter_time = time.time()

            # for running with a subset of the data
            if data_interval == None:
                line = LinearNDInterpolator( self.input_data, cl )
            else:
                line = LinearNDInterpolator( self.input_data[::data_interval], cl[::data_interval])

            line_holder[self.class_names[i]] = line

            time_print = time.time()-start_time
            if i == 0:
                len_classes = len(self.class_ids)
                print( "Time to fit %s classifiers ~ %.3f\n"%(len_classes, time_print*len_classes) )
            print("LinearNDInterpolator class %s -- current time: %.3f"%(i,time_print) )

        self._interpolators_["LinearNDInterpolator"] = line_holder
        print("Done training LinearNDInterpolator.")

    def rbf_interpolator(self,):
        start_time = time.time()
        print("foobar")

    def gaussian_process_classifier():
        start_time = time.time()
        print("foo")


    def return_probs(self, test_input, all_probs = False):
        # all_probs = True returns N x M matrix where
        # N rows = num test points, M columns = num classes

        probs = []
        
        if not self._interpolators_: # if empty
            raise Exception("\n\nNo trained interpolators exist.")

        for interp_tuple in self._interpolators_:
            interp = interp_tuple[1] # 0 key, 1 mapped object
            # - each interpolator is on a different class
            probs.append( interp( test_input ) ) # for all test points do one interpolator
        # --- 1st array in probs = [ <class 0 prob on 1st test point>, <class 0 prob on 2nd test point>,... ]
        #  to get a row where each element is from a diff class -> transpose probs
        probs = np.array(probs).T

        if all_probs:
            return probs
        else:
            return np.max(probs, axis = 1)


    def return_class_predictions(self, test_input, return_probs = False):

        probs = self.return_probs(test_input, all_probs = True)

        pred_class_ids = np.argmax( probs, axis = 1 )

        if return_probs:
            return pred_class_ids, probs
        else:
            return pred_class_ids


    def get_rnd_test_inputs(self, N):
        # produce randomly sampled 'test' inputs inside domain of input_data
        # ++ need ability to specify rng of axis

        num_axis = len(self.input_data[0])

        # find max and min in each axis
        a_max = []; a_min = []
        for i in range(num_axis):
            a_max.append( max(self.input_data.T[i]) )
            a_min.append( min(self.input_data.T[i]) )
        # sample N points between max & min in each axis
        axis_rnd_points = []
        for i in range(num_axis):
            r = np.random.uniform( low = a_min[i], high = a_max[i], size = N )
            # this reshape is necessary to concatenate
            axis_rnd_points.append( r[:,np.newaxis] )

        # now put the random points back together with same shape as input_data
        rnd_test_points = np.concatenate( axis_rnd_points, axis = 1  )

        return rnd_test_points