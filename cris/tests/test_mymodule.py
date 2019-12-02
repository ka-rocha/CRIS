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
import os

from cris.data import TableData
from cris.classify import Classifier
from cris.regress import Regressor
from cris.sample import Sampler

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

class TestCRIS(unittest.TestCase):

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
