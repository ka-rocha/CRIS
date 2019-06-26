import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


class TableData():
    """
    TableData
    =========

    For managing data sets used for classification and regression.
    Required data object for Classifier and Regressor.
    """

    def __init__(self, table_paths, input_cols, output_cols, class_col, ignore_lines = 0, subset_interval = None):
        # +++ probably want to add some functionality for direct load of df

        self.df_list = [] # data file list
        self.df_index_keys = [] # index keys (for rows in full_data)

        self._files_ = table_paths
        self.class_col = class_col # assumed to be one column name + string

        # read in all data files and add to df_list
        print( "Reading in data from %i file(s)."%(len(table_paths)) )
        for num, path in enumerate(table_paths):
            print(path)
            df = pd.read_csv( path, header=ignore_lines, delim_whitespace=True )
            self.df_list.append( df )
            self.df_index_keys.append( 'df' + str(num) )
        print( "Finished reading data.\n" )

        # df_index_keys setting index of Nth file the data is from with 'dfN'.
        self.full_data = pd.concat( self.df_list, join='outer',
                                   ignore_index = False, keys= self.df_index_keys)

        if subset_interval is not None:
            len_original_data = len(self.full_data)
            self.full_data = self.full_data.iloc[np.arange(0, len(self.full_data), subset_interval),:]
            print("--Using Subset--")
            print( "%.2f percent of total data set."%( len(self.full_data)/len_original_data * 100 ) )

        print( "Total number of data points: %i\n"%len(self.full_data) )

        # column names in fully concatenated data
        self.col_names = np.array(self.full_data.columns)

        # input and output data
        self.input_  = self.full_data[input_cols]
        self.output_ = self.full_data[output_cols]

        # --------- classification variables ---------
        self.class_names = np.unique( self.full_data[class_col] )
        self.num_classes = len(self.class_names)
        self.class_ids = np.arange( 0, self.num_classes, 1, dtype=int)

        print("Input columns: %s"%(len(input_cols)) )
        print("Output columns: %s"%(len(output_cols)) )
        print("Unique classes found in %s: %s"%(class_col, self.num_classes) )

        # mapping dictionary - forward & backward
        self.class_id_mapping = dict()
        for i in range(self.num_classes):
            self.class_id_mapping[i] = self.class_names[i]
            self.class_id_mapping[ self.class_names[i] ] = i

        # single column of classes replaced with class_id
        self.all_classes = self.full_data[class_col].values
        self.classes_to_ids = []
        for cl in self.all_classes:
            self.classes_to_ids.append( self.class_id_mapping[cl] )


        # --------- regression variables ---------

        # make input and output dicts for each class
        self.regr_inputs_ = dict()
        self.regr_outputs_ = dict()
        for cls_name in self.class_names:
            rows_where_class = np.where( self.output_[self.class_col] == cls_name )
            self.regr_inputs_[cls_name] =  self.input_.iloc[rows_where_class]
            self.regr_outputs_[cls_name] = self.output_.iloc[rows_where_class]

        print("\nFinding values to regress:\n")
        print("Num output(s) \t Class Name")

        self.regr_names = self.output_.columns
        self.num_outputs_to_regr = []
        self.regr_dfs_per_class = dict()
        # for each class - find columns which can be converted to floats
        for i, tuple_output in enumerate( self.regr_outputs_.items() ):
            output_per_class = tuple_output[1] # 0 returns key; 1 returns data

            cols_with_float = []
            bad_cols = []
            # go through first row and try to convert each element to float
            for col_num, val in enumerate( output_per_class.iloc[0,:]):
                try:
                    var = float(val) # if fails -> straight to except (skips next line)
                    cols_with_float.append( col_num )
                except:
                    bad_cols.append( col_num )

            print( "%7i \t '%s'"%( len(cols_with_float), self.class_names[i] )  )

            self.num_outputs_to_regr.append( len(cols_with_float) )

            regression_vals_df = output_per_class.iloc[:,cols_with_float]
            # 'regression_vals_df' has the regression data frame for a given class

            # if regressable elements - link class with the df of valid floats
            # if no elements to regress - append np.nan
            if len(cols_with_float) <= 0:
                self.regr_dfs_per_class[self.class_names[i]] = np.nan
            else:
                self.regr_dfs_per_class[self.class_names[i]] = regression_vals_df


    def get_full_data(self, return_df = False):
        """Get all data contained in DataTable object.

        Parameters
        ----------
        return_df: bool, optional
            If True, return the pandas data frame object.
            If False, return a numpy array. (default)

        Returns
        -------
        full_data: ndarray
            An array containing all data from loaded files.
        """
        if return_df:
            return self.full_data
        else:
            return np.array(self.full_data)


    def get_input_data(self,return_df=False):
        """Get all input data.

        Parameters
        ----------
        return_df: bool, optional
            If True, return the pandas data frame object.
            If False, return a numpy array. (default)

        Returns
        -------
        input_data: ndarray
            An array containing all input data.
        """
        if return_df:
            return self.input_
        else:
            return np.array(self.input_)

    def get_output_data(self,return_df=False):
        if return_df:
            return self.output_
        else:
            return np.array(self.output_)


    def get_class_ids(self,):
        return self.class_ids

    def get_classes_to_ids(self,):
        """ Dictionary of class names to their respective IDs."""
        return self.classes_to_ids

    def get_class_data( self,):
        # Return list of N arrays. Each array corresponds to one class and
        # has 1 for that class and 0 for all other classes.
        cls = np.array( self.output_[self.class_col] )

        # create a different array for each class - one against all
        all_classifiers = []
        for i in range(self.num_classes):
            # 1 for True, 0 for False
            where_class_is = np.where( cls == self.class_names[i], 1, 0 )
            all_classifiers.append( np.concatenate(where_class_is, axis=None) )

        return np.array(all_classifiers)

    def get_classes_to_ids(self,):
        return np.array( self.classes_to_ids )

    def get_class_names(self,):
        return np.array( self.class_names )



    def get_regr_input_data(self,):
        """Dictionary"""
        return self.regr_inputs_

    def get_regr_output_data(self,):
        """Dictionary"""
        return self.regr_outputs_

    def get_regr_sorted_output_data(self,):
        """Dictionary"""
        return self.regr_dfs_per_class



    def plot_class_data(self, which_val = 0):
        # right now the get_class_data does not behave similarly to
        # this function and I think it should have a diff name or something

        # this works for 3D.....
        full_data = self.full_data
        input_data = self.input_

        first_axis  = 'log10(M_1i)(Msun)' # args
        second_axis = 'P_i(days)'         # args
        third_axis  = 'metallicity'       # args
        #which_val   = 0 # index of slice value in 3rd axis

        colors = ['#EC6666',
                  '#90A245',
                  '#F5C258',
                  '#1668E8',
                  '#473335',
                  '#98C0CB']

        color_dict = {0:colors[0],
                      1:colors[1],
                      2:colors[2],
                      3:colors[3],
                      4:colors[4],
                      5:colors[5] }
        ## MAKE LEGEND
        legend_elements = []
        for i, name in enumerate(self.class_names):
            legend_elements.append( Line2D([], [], marker='s', color=color_dict[i],
                                           label=name, linestyle='None', markersize=8)  )
        #--------------

        # Specify value of other '3rd' axis not in 2D plot
        slice_value = np.unique(full_data[third_axis])[which_val]
        where_to_slice = full_data[third_axis] == slice_value # boolean data frame

        # Find all indicies (rows) that have the slice value
        what_index = np.where( where_to_slice == True )[0]
        IDs = np.array( self.classes_to_ids )
        data_to_plot = full_data[where_to_slice]

        # class IDs (ints 0->n) to color code
        class_to_colors = [ color_dict[val] for val in IDs[what_index] ]

        plt.figure( figsize=(4,5), dpi=120 )
        plt.title( third_axis + '= %f'%(slice_value) )
        plt.scatter( data_to_plot[first_axis],
                     data_to_plot[second_axis],
                     c = class_to_colors,
                     cmap = None, s=12, marker = 's')

        plt.xlabel(first_axis); plt.ylabel(second_axis)
        plt.legend( handles = legend_elements, bbox_to_anchor = (1.03, 1.02))
        plt.show()
