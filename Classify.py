import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from scipy.interpolate import LinearNDInterpolator
import time


class Classify():

    def __init__(self, table_data, method = "linear"):

        self.method = method
        self.table_data = table_data

        self.classes = table_data.get_classes()
        self.class_data = table_data.get_class_data()
        self.input_data = table_data.get_input_data()
        self.interpolators = []

        start_time = time.time()
        for i, cl in enumerate(self.class_data):
            iter_time = time.time()

            line = LinearNDInterpolator( self.input_data[0::5], cl[0::5])
            #line = LinearNDInterpolator( self.input_data, cl )
            self.interpolators.append( line )

            time_print = time.time()-start_time
            if i == 0:
                len_classes = len(self.classes)
                print( "Time to fit %s classifiers ~ %.3f\n"%(len_classes, time_print*len_classes) )
            print("LinearNDInterpolator class %s -- time: %.3f"%(i,time_print) )

        print("Done")


    def return_probs(self, test_input, all_probs = False):
        # all_probs = True returns N x M matrix where
        # N rows = num test points, M columns = num classes

        probs = []
        classes = self.table_data.get_classes()
        class_data = self.table_data.get_class_data()

        for interp in self.interpolators:
            # for LinearNDInterpolator - each interpolator is on a different class
            probs.append( interp( test_input ) ) # for all test points do one interpolator
        # --- 1st array in probs = [ <class 0 prob on 1st test point>, <class 0 prob on 2nd test point>,... ]
        #  to get a row where each element is from a diff class -> transpose probs
        probs = np.array(probs).T

        if all_probs:
            return probs
        else:
            return np.max(probs, axis = 1)

    def return_class_predictions(self, test_input):

        probs = self.return_probs(test_input, all_probs = True)

        pred_class_ids = np.argmax( probs, axis = 1 )
        return pred_class_ids


    def get_rnd_test_inputs(self, N):
        # produce randomly sampled 'test' inputs inside domain of input_data

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
