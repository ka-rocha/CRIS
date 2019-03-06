import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


class TabelData():
    """
    TableData
    =========

    For managing data sets used for classification and regression.
    Required data object for Classifier and Regressor.

    """

    def __init__(self, table_paths, input_cols, output_cols, class_col, ignore_lines = 0):
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

        # column names in fully concatenated data
        self.col_names = np.array(self.full_data.columns)

        # input and output data
        self.input_  = self.full_data[input_cols]
        self.output_ = self.full_data[output_cols]

        # ---- classification variables ----
        self.class_names = np.unique( self.full_data[class_col] )
        self.num_classes = len(self.class_names)
        self.class_ids = np.arange( 0, self.num_classes, 1, dtype=int)

        print("Input columns: %s"%(len(input_cols)) )
        print("Output columns: %s"%(len(output_cols)) )
        print("Unique classes found in %s: %s"%(class_col, self.num_classes) )

        # ---- regression variables ----


        # mapping dictionary - forward & backward
        self.class_id_mapping = dict()
        for i in range(self.num_classes):
            self.class_id_mapping[i] = self.class_names[i]
            self.class_id_mapping[ self.class_names[i] ] = i

        # class column replaced with class_id
        self.all_classes = self.full_data[class_col].values
        self.classes_to_ids = []
        for cl in self.all_classes:
            self.classes_to_ids.append( self.class_id_mapping[cl] )


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


    def get_classes(self,):
        return self.class_ids

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



    def plot_class_data(self,):
        # right now the get_class_data does not behave similarly to
        # this function and I think it should have a diff name or something

        # this works for 3D.....

        full_data = self.full_data
        input_data = self.input_

        first_axis  = 'log10(M_1i)(Msun)' # args
        second_axis = 'P_i(days)'         # args
        third_axis  = 'metallicity'       # args
        which_val   = 0 # index of slice value in 3rd axis

        colors = ['#8ed6ff',
                  '#e88a1a',
                  '#729d39',
                  '#ffc0c2',
                  '#0d627a',
                  '#ffe867']
        color_dict = {0:colors[0],
                      1:colors[1],
                      2:colors[2],
                      3:colors[3],
                      4:colors[4],
                      5:colors[5] }
        ## MAKE LEGEND
        legend_elements = []
        for i, name in enumerate(table_object.class_names):
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
