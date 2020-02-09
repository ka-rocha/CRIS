# -*- coding: utf-8 -*-
# Copyright (C) YOUR NAME (2019)
#
# This file is part of YOURPACKAGE.
#
# YOURPACKAGE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# YOURPACKAGE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with YOURPACKAGE.  If not, see <http://www.gnu.org/licenses/>.

"""Unit test for YOURPACKAGE.YOURMODULE classes
"""
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from cris.data import TableData
from cris.classify import Classifier
from cris.regress import Regressor
from cris.sample import Sampler

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

class Test_CRIS_TableData(unittest.TestCase):

    def test_init_TableData(self):
        files = None
        input_cols = ['input_1', 'input_2' ]
        output_cols = ['class', 'output_1']
        class_col_name = 'class'
        loaded_data = pd.read_csv(os.path.join(TEST_DATA_DIR, "synth_data.dat"))
        table_object = TableData( files, input_cols, output_cols, class_col_name,
                                  my_DataFrame = loaded_data, verbose=False, n_neighbors=[2,3])
        return table_object

    def test_data_return_methods(self):
        table_object = self.test_init_TableData()
        binary_data = table_object.get_binary_mapping_per_class()
        self.assertTrue( np.shape(binary_data) == (4,100) ) # each row is unique class

        all_class_data = table_object.get_class_data(what_data="full")
        self.assertTrue( len(all_class_data)==5 ) # 5 returns
        self.assertTrue( len(all_class_data[0])==100 ) # len(class_col) = 100

        all_input_data = table_object.get_data(what_data='input')
        self.assertTrue( np.shape(all_input_data) == (100,2) ) # correct shape

        n_neighbors = 3
        where_nearest = table_object.find_n_neighbors( all_input_data, [n_neighbors] )
        self.assertTrue( np.shape(where_nearest[n_neighbors]) == (100,n_neighbors) ) # correct shape

        my_info = table_object.get_info()

        all_regr_data = table_object.get_regr_data(what_data="full")
        regr_output = all_regr_data[2]
        self.assertTrue( len(regr_output["C"]) == 4 ) # only 4 points in class C

    def test_plotting_methods(self):
        table_object = self.test_init_TableData()
        fig, subs = plt.subplots( nrows=1, ncols=2 )
        table_object.make_class_data_plot( fig, subs[0], ["input_1", "input_2"] )
        table_object.make_class_data_plot( fig, subs[1], ["input_2", "input_1"] )


class Test_CRIS_Classifier(unittest.TestCase):
    # The only methods not covered:
    #   Classifier.make_cv_plot_data -> wrapper on cross_validate
    #   Classifier.make_max_cls_plot -> simply doesn't work currently
    def get_TableData(self):
        files = None
        input_cols = ['input_1', 'input_2' ]
        output_cols = ['class', 'output_1']
        class_col_name = 'class'
        loaded_data = pd.read_csv(os.path.join(TEST_DATA_DIR, "synth_data.dat"))
        table_object = TableData( files, input_cols, output_cols, class_col_name,
                                  my_DataFrame = loaded_data, verbose=False, n_neighbors=[2,3])
        return table_object

    def test_init_Classifier(self):
        table_object = self.get_TableData()
        classifier_obj = Classifier( table_object )
        return classifier_obj

    def test_training(self):
        cls_obj = self.test_init_Classifier()
        cls_obj.train( "linear"  )
        cls_obj.train( "linear", di = np.arange(0,100,2)  ) # testing di
        cls_obj.train( "gp", **{"n_restarts":10} )
        cls_obj.train_everything( ["rbf", "gp"] ) # all interpolators

    def test_get_rnd_test_inputs(self):
        cls_obj = self.test_init_Classifier()
        my_points1 = cls_obj.get_rnd_test_inputs(10)
        self.assertTrue( np.shape(my_points1) == (10,2) )
        my_points2 = cls_obj.get_rnd_test_inputs(10, other_rng={1:(11,12)})
        self.assertTrue( (my_points2.T[1] > 10).all() ) # specifying other range to sample
        return my_points1

    def test_cross_validation(self):
        cls_obj = self.test_init_Classifier()
        p_corr, t = cls_obj.cross_validate( ["linear"], 0.2 )
        self.assertAlmostEqual( p_corr[0], 73, delta=25, msg="CrossVal linear, alpha 0.2" )
        # Mean = 73, Standard Deviation = 6 for 1000 iterations

        p_corr, t = cls_obj.cross_validate( ["rbf"], 0.2 ) # should just run
        p_corr, t = cls_obj.cross_validate( ["gp"], 0.6 ) # should just run
        # sometimes errors for low alpha, not enough classes

    def test_return_probs_and_get_class_predictions(self):
        cls_obj = self.test_init_Classifier()
        cls_list = ["linear", "rbf", "gp"]
        cls_obj.train_everything( cls_list )
        my_points = self.test_get_rnd_test_inputs()
        for cls in cls_list:
            probs, where_not_nan = cls_obj.return_probs( cls, my_points)
            self.assertTrue( np.sum(probs) == len(my_points) ) # all probs sum to one
        pred_class_ids, probs, where_not_nan = cls_obj.get_class_predictions("gp", my_points)
        self.assertTrue( len(pred_class_ids) == len(where_not_nan) )

    def test_get_classifier_name_to_key(self):
        cls_obj = self.test_init_Classifier()
        self.assertTrue( cls_obj.get_classifier_name_to_key("lin") == "LinearNDInterpolator" )
        self.assertTrue( cls_obj.get_classifier_name_to_key("rbf") == "RBF" )
        self.assertTrue( cls_obj.get_classifier_name_to_key("gp") == "GaussianProcessClassifier" )

    def test_get_cross_val_data(self):
        cls_obj = self.test_init_Classifier()
        sorted_int_vals, input_data, output_data = cls_obj.get_cross_val_data( 0.5 )
        self.assertTrue( len(input_data) == len(output_data) == len(sorted_int_vals) == (0.5 * 100) )


class Test_CRIS_Regressor(unittest.TestCase):

    def get_TableData(self):
        files = None
        input_cols = ['input_1', 'input_2' ]
        output_cols = ['class', 'output_1']
        class_col_name = 'class'
        loaded_data = pd.read_csv(os.path.join(TEST_DATA_DIR, "synth_data.dat"))
        table_object = TableData( files, input_cols, output_cols, class_col_name,
                                  my_DataFrame = loaded_data, verbose=False, n_neighbors=[2,3])
        return table_object

    def test_init_Regressor(self):
        table_object = self.get_TableData()
        regr_obj = Regressor( table_object )
        return regr_obj

    def test_training(self):
        regr_obj = self.test_init_Regressor()
        regr_obj.train( "rbf", ["A"], ["output_1"], verbose=True )
        regr_obj.train( "gp", ["B", "D"], ["output_1"], verbose=True )
        regr_obj.train( "linear", ["A", "B", "D"], ["output_1"], verbose=True )
        regr_obj.train_everything( ["rbf"], verbose=False )
        return regr_obj

    def test_get_predictions(self):
        regr_obj = self.test_training()
        preds = regr_obj.get_predictions( ["rbf"], ["A"], ["output_1"], np.array([[2,3]]) )
        self.assertTrue( preds["RBF"]["A"]["output_1"][0] == -0.5625707225289136 )

    def test_get_regressor_name_to_key(self):
        regr_obj = self.test_init_Regressor()
        self.assertTrue( regr_obj.get_regressor_name_to_key("rbf") == "RBF" )
        self.assertTrue( regr_obj.get_regressor_name_to_key("gp") == "GaussianProcessRegressor" )
        self.assertTrue( regr_obj.get_regressor_name_to_key("lin") == "LinearNDInterpolator" )

    def test_show_structure(self):
        regr_obj = self.test_init_Regressor()
        regr_obj.show_structure()

    def test_get_cross_val_data(self):
        regr_obj = self.test_init_Regressor()
        test_input, test_output, indicies = regr_obj.get_cross_val_data( "A", "output_1", 0.5 )
        self.assertTrue( len(indicies) == 6 )
        test_input, test_output, indicies = regr_obj.get_cross_val_data( "B", "output_1", 0.5 )
        self.assertTrue( len(indicies) == 19 )

    def test_cross_validate(self):
        regr_obj = self.test_init_Regressor()
        p_diff, diffs = regr_obj.cross_validate( "rbf", "A", "output_1", 0.25, verbose=True)

    def test_get_max_APC_val(self):
        regr_obj = self.test_training()
        regr_obj.get_max_APC_val( "rbf", "D", np.array([[0,1]]) )

    def test_get_rnd_test_inputs(self):
        regr_obj = self.test_init_Regressor()
        test_in = regr_obj.get_rnd_test_inputs( "A", 10 )
        self.assertTrue( test_in.shape == (10,2) )
        other_test_in = regr_obj.get_rnd_test_inputs( "A", 10, other_rng={0:[-2,-1]} )
        self.assertTrue( (other_test_in[:,0] < 0).all() )

    def test_plot_regr_data(self):
        regr_obj = self.test_init_Regressor()
        regr_obj.plot_regr_data("A")




class Test_CRIS(unittest.TestCase):

    def get_table_object(self):
        files = None
        input_cols = ['input_1', 'input_2' ]
        output_cols = ['class', 'output_1']
        class_col_name = 'class'
        loaded_data = pd.read_csv(os.path.join(TEST_DATA_DIR, "synth_data.dat"))
        table_object = TableData( files, input_cols, output_cols, class_col_name,
                                  my_DataFrame = loaded_data, verbose=False, n_neighbors=[2])
        return table_object

    def get_classifier_and_regressor_objects(self):
        var = self.get_table_object()
        classifier_obj = Classifier( var )
        classifier_obj.train_everything( ['linear', 'rbf', 'gp'] )
        regressor_obj = Regressor( var )
        regressor_obj.train("rbf", ["A","C"], None, verbose=False)
        return classifier_obj, regressor_obj

    def test_tabledata_using_synthetic_data(self):
        var = self.get_table_object()
        self.assertTrue( var.num_classes == 4 )

    def test_classification(self):
        var = self.get_table_object()
        classifier_obj = Classifier( var )
        classifier_obj.train_everything( ['linear', 'rbf', 'gp'] )
        probs, where_not_nan = classifier_obj.return_probs( 'linear', np.array([[1,1.5]]) )
        self.assertAlmostEqual(  probs[0,2], 0.25, delta=1e-10, msg="class C probability" )
        self.assertAlmostEqual(  probs[0,3], 0.75, delta=1e-10, msg="class D probability" )

    def test_regression(self):
        var = self.get_table_object()
        regressor_obj = Regressor( var )
        regressor_obj.train("rbf", ["A","C"], None, verbose=False)
        random_inputs = regressor_obj.get_rnd_test_inputs("A", 4)
        preds = regressor_obj.get_predictions(["rbf"], ["A", "C"], ["output_1"], random_inputs )
        self.assertTrue( np.sum( preds["RBF"]["A"]["output_1"] < 0) == 4 ) # all 4 values less than 0
        self.assertTrue( np.sum( preds["RBF"]["C"]["output_1"] > 0) == 4 ) # all 4 values greater than 0

    def test_sampler(self):
        classifier_obj, regressor_obj = self.get_classifier_and_regressor_objects()
        sampler_obj = Sampler( classifier = classifier_obj, regressor= regressor_obj )

        steps, acc, rej = sampler_obj.run_MCMC(20, 0.5, [[0,0]], sampler_obj.classifier_target_dist, 'rbf'  )

        chain_steps, T_list = sampler_obj.run_PTMCMC( 15, 200, [0,0],
                                sampler_obj.classifier_target_dist,  "rbf",  alpha=0.3, verbose=True)
        self.assertTrue( len(chain_steps) == len(T_list) )

        points, kapp = sampler_obj.get_proposed_points( chain_steps[len(T_list)-1], 20, 25.5,
                                                        verbose=True)
        self.assertTrue( len(points) == 20 )
        self.assertTrue( kapp > 0 )
        self.assertTrue( np.sum(abs(points.T[0]) > 3)==0 ) # no points outside domain of data
        self.assertTrue( np.sum(abs(points.T[1]) > 3)==0 ) # no points outside domain of data



if __name__ == '__main__':
    unittest.main()
