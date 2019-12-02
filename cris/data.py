import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import math
import time

from sklearn.neighbors import NearestNeighbors

def calc_avg_dist(data, n_neighbors, neighbor=None):
    """Given data, return the average distance to the
    n_neigh nearest neighbors in the data set.
    (NearestNeighbors from sklearn.neighbors)

    Params
    ------
    data : ndarray
        Data to train the NearestNeighbors class on.
    n_neighbors : int
        Number of neighbors to use when finding average distance.
    neighbor : instance of NearestNeightbors class
        For passing your own object.

    Returns
    -------
    avg_dist : array
        The average distance between nearest neighbors.
    g_indi : array
        Indicies that correspond to the nearest neighbors.
    """
    if neighbor:
        neigh = neighbor
    else:
        neigh = NearestNeighbors()
    neigh.fit(data)

    # since we pass the original data, the first nearest point is itself
    # if we want 2 nearest neighbors, we ask for 3 because the first will always be 0
    dist, indicies = neigh.kneighbors( data, n_neighbors=(n_neighbors+1) )

    g_dist = (dist.T[1:len(dist)]).T
    g_indi = (indicies.T[1:len(indicies)]).T
    avg_dist =  np.mean(g_dist, axis = 1)
    return avg_dist, g_indi


def calc_avg_p_change( data, where_nearest_neighbors ):
    """Calculate the average percent or fractional change in a given data set.

    Parameters
    ----------
    data : ndarray
        Data set to calculate percent change.
    where_nearest_neighbors : dict
        Indicies in data for the n nearest neighbors in input space.

    Returns
    -------
    avg_p_change_holder : ndarray
        Each element conatins the average percent change for a given number of neighbors.
    """
    where_zero = np.where(data==0)[0]
    if len(where_zero)>0: # can't calculate percent change from 0
        return None
    else:
        pass

    avg_p_change_holder = []
    for n_, indicies in where_nearest_neighbors.items():
        diff_holder = []
        for i in range(n_):
            nearest_data = data[ indicies.T[i] ]
            diff_holder.append( data - nearest_data )

        diffs = np.array(diff_holder).T
        avg_diffs = np.mean( diffs, axis=1 )
        avg_p_change = abs(avg_diffs)/data

        avg_p_change_holder.append( avg_p_change )

    return np.array(avg_p_change_holder)




class TableData():
    """
    TableData
    =========

    For managing data sets used for classification and regression.

        Reads tables of simulation data where a single row represents one
        simulation. Each column in a row represents different inputs
        (initial conditions) and outputs (result, continuous variables).
        If using multiple files, each file is assumed to have the same
        columns. You may also directly load a pandas DataFrame instead of
        reading in files.

        Example data structure expected in files or pandas DataFrame:
        0  input_1  input_2  outcome  output_1   output_2  output_3 ...
        1    1.5      2.6      "A"       100       0.19       -     ...
        2    1.5      3.0      "B"        -          -        -     ...
        3    2.0      2.6      "C"        -          -        6     ...
        ...

        The above table has dashes '-' in output columns to indicate Nan values.
        This may occur if different classes have fundamentally different outputs.

    Parameters
    ----------
    table_paths : list
        List of file paths to read in as data.
    input_cols : list
        List of names of the columns which will be considered 'input'.
    output_cols : list
        List of names of the columns which will be considered 'output'.
    class_col_name : str
        Name of column which contains classification data.
    my_DataFrame : pandas DataFrame object, optional
        If given, use this instead of file paths.
    ignore_lines : int, optional
        Number of lines to ignore if files have a header.
    omit_vals : list, optional
        Numerical values that you wish to omit from the data set.
        If a row contains the value, the entire row is removed.
        For example you way want to omit all rows if they contain "-1".
    omit_cols : list, optional
        Column names that you wish to omit from the data set.
    subset_interval : array, optional
        Use some subset of the data files being loaded in.
        An array with integers indicating the rows that will be kept.
    my_colors : list, optional
        Colors to use for classification plots.
    n_neighbors : list, optional
        List of integers that set the number of neighbors to use to
        calculate average distances. (default None)
    neighbor : instance of NearestNeighbors class, optional
        To use for average distances. See function 'calc_avg_dist()'.
    verbose : bool, optional
        Print statements with extra info.
    **kwargs : dict, optional
        Kwargs used in the pandas function 'read_csv()'.

    Attributes
    ----------

    Methods
    -------
    """

    def __vb_helper(self, verbose_bool, info_string):
        """Help clean up verbose print statements and storing info.
        By default, store the passed info_string regardless of verbose_bool."""
        if verbose_bool: print(info_string)
        self._for_info_.append(info_string)


    def __init__(self, table_paths, input_cols, output_cols, class_col_name, \
                    my_DataFrame = None, omit_vals=None, omit_cols=None,
                    subset_interval=None, verbose=False, my_colors=None, \
                    neighbor=None, n_neighbors=None, **kwargs ):

        start_time = time.time()
        self._for_info_ = []  # storing info strings

        # --------- data pre-processing ---------
        self._df_list_ = [] # data frame list
        self._df_index_keys_ = [] # index keys (for rows in _full_data_)
        self._files_ = table_paths
        self.class_col_name = class_col_name # assumed to be one column name + string

        if isinstance( my_DataFrame, pd.DataFrame ):
            self.__vb_helper( verbose, "Using loaded Pandas DataFrame")
            self._full_data_ = my_DataFrame
        else:
            # Read in all data files and add to _df_list_
            info_str_01 = "Reading in data from {0} file(s).".format(len(table_paths))
            self.__vb_helper(verbose, info_str_01)

            for num, path in enumerate(table_paths):
                info_str_02 = "\t'{0}'".format(path)
                self.__vb_helper(verbose, info_str_02)

                df = pd.read_csv( path, **kwargs )
                self._df_list_.append( df )
                self._df_index_keys_.append( 'df' + str(num) )

            info_str_03 = "Finished reading data.\n"
            self.__vb_helper(verbose, info_str_03)

            # _df_index_keys_ setting index of Nth file the data is from with 'dfN'.
            self._full_data_ = pd.concat( self._df_list_, join='outer',
                                          ignore_index=False, keys=self._df_index_keys_,\
                                          sort=True)

        # remove rows and columns with unwanted data
        if omit_vals is not None:
            ct = 0 # counter
            for j, val in enumerate(omit_vals):
                row_where_omit_val = np.where( self._full_data_.values == val )[0]
                df_keys_for_omit = self._full_data_.index[row_where_omit_val]
                self._full_data_ = self._full_data_.drop( df_keys_for_omit, axis=0 )
                ct += len( row_where_omit_val )

                info_str_04 = " - Removed {0} rows containing: {1}".format(len(row_where_omit_val), omit_vals[j])
                self.__vb_helper(verbose, info_str_04)
            info_str_05 = "Removed a total of {0} rows.".format(ct)
            self.__vb_helper(verbose, info_str_05)

        # remove entire columns
        if omit_cols is not None:
            self._full_data_ = self._full_data_.drop( columns = omit_cols )

            info_str_06 = "Removed columns: {0}".format(omit_cols)
            self.__vb_helper(verbose, info_str_06)

        # use a subset of the original data table
        if subset_interval is not None:
            len_original_data = len(self._full_data_)
            self._full_data_ = self._full_data_.iloc[subset_interval,:]
            info_str_07 = "--Using Subset--\n" + "{0} percent of total data set.".format(len(self._full_data_)/len_original_data*100)
            self.__vb_helper(verbose, info_str_07)

        info_str_08 = "Total number of data points: {0}\n".format(len(self._full_data_))
        self.__vb_helper(verbose, info_str_08)

        # column names in fully concatenated data
        self.col_names = np.array(self._full_data_.columns)

        # input and output data
        for usr_input in [input_cols, output_cols]:
            for a_name in usr_input:
                if a_name not in self.col_names:
                    info_str_09 = " !!! No columns with name '{0}' ".format(a_name)
                    self.__vb_helper(verbose, info_str_09)
        input_cols = [ i for i in input_cols if i in self.col_names ]
        output_cols = [ i for i in output_cols if i in self.col_names ]
        self._input_  = self._full_data_[input_cols]
        self._output_ = self._full_data_[output_cols]

        # --------- classification variables ---------
        try: # TODO: this catch is not working correctly for some reason
            self._unique_class_keys_ = np.unique( self._full_data_[class_col_name] )
        except KeyError():
            info_str_10 = "'{0}' not in {1}".format(class_col_name, np.array(self._full_data_.keys()))
            self.__vb_helper(verbose, info_str_10)
        self.num_classes = len(self._unique_class_keys_)
        self.class_ids = np.arange( 0, self.num_classes, 1, dtype=int)

        info_str_11 = "Input columns: {0}".format(len(input_cols))  + "\n" \
                      + "Output columns: {0}".format(len(output_cols)) + "\n" \
                      + "Unique classes found in '{0}': {1}".format(class_col_name, self.num_classes)
        self.__vb_helper(verbose, info_str_11)

        # mapping dict - forward & backward
        self._class_id_mapping_ = dict()
        for i in range(self.num_classes):
            self._class_id_mapping_[i] = self._unique_class_keys_[i]
            self._class_id_mapping_[ self._unique_class_keys_[i] ] = i

        # classification column replaced with class_id
        self._class_col_ = self._full_data_[class_col_name].values
        self._class_col_to_ids_ = []
        for cl in self._class_col_:
            self._class_col_to_ids_.append( self._class_id_mapping_[cl] )

        if my_colors is not None:
            self._class_colors_ = my_colors
            self.__vb_helper(verbose, "Using custom class colors.")
        else:
            self._class_colors_ = ['#EC6666', '#90A245', '#F5C258', \
                                   '#1668E8', '#473335', '#98C0CB', \
                                   'C0', 'C1', 'C2', 'C3', 'C4', \
                                   'C5', 'C6', 'C7', 'C8']
            self.__vb_helper(verbose, "Using default class colors.")

        # --------- regression variables ---------

        # make input and output dicts for each class
        self._regr_inputs_ = dict()
        self._regr_outputs_ = dict()
        for cls_name in self._unique_class_keys_:
            rows_where_class = np.where( self._output_[class_col_name] == cls_name )
            self._regr_inputs_[cls_name] =  self._input_.iloc[rows_where_class]
            self._regr_outputs_[cls_name] = self._output_.iloc[rows_where_class]

        info_str_12 = "\nFinding values to regress:\n" + "Num output(s) \t Class Name"
        self.__vb_helper(verbose, info_str_12)

        # find valid regression data and link it to a class
        self.regr_names = self._output_.columns
        self.num_outputs_to_regr = []
        self._regr_dfs_per_class_ = dict()
        # for each class - find columns which can be converted to floats
        for i, tuple_output in enumerate( self._regr_outputs_.items() ):
            output_per_class = tuple_output[1] # 0 returns key; 1 returns data

            cols_with_float = []
            bad_cols = []
            # go through first row and try to convert each element to float - also check if nan
            for col_num, val in enumerate( output_per_class.iloc[0,:]):
                try:
                    var = float(val) # if fails -> straight to except (skips next line)
                    if math.isnan(var):
                        bad_cols.append( col_num )
                    else:
                        cols_with_float.append( col_num )
                except:
                    bad_cols.append( col_num )

            info_str_13 = "%7i \t '%s'"%( len(cols_with_float), self._unique_class_keys_[i] )
            self.__vb_helper(verbose, info_str_13)

            self.num_outputs_to_regr.append( len(cols_with_float) )

            regression_vals_df = output_per_class.iloc[:,cols_with_float]
            # 'regression_vals_df' has the regression data frame for a given class

            # if regressable elements - link class with the df of valid floats, else - None
            if cols_with_float:
                self._regr_dfs_per_class_[self._unique_class_keys_[i]] = regression_vals_df
            else:
                self._regr_dfs_per_class_[self._unique_class_keys_[i]] = np.nan


        # take Nearest Neighbors differences
        if n_neighbors is not None:
            info_str_14 = "\nCalculate Average Distances & Average Percent Change"
            self.__vb_helper(verbose, info_str_14)

            self._n_neighbors_ = [int(i) for i in n_neighbors]
            self._avg_dist_dfs_per_class_ = dict() # stores the average distances

            # We need distances in input space, then compare %diff in output space
            for key, val in self._regr_inputs_.items():
                # key is the class, val is input DataFrame or np.nan
                self._avg_dist_dfs_per_class_[key] = dict()
                regr_data = self._regr_dfs_per_class_[key]

                if isinstance(regr_data, pd.DataFrame):
                    self.__vb_helper(verbose, "class: '{0}'".format(key))

                    # find nearest neighbors in input space
                    where_nearest_neighbors = dict()
                    for n_ in self._n_neighbors_:
                        avg_dist, indicies = calc_avg_dist(val.values, n_, neighbor=neighbor)
                        where_nearest_neighbors[n_] = indicies

                    for _key, _val in regr_data.items(): # regression data
                        data = np.array(_val.values, dtype=float)
                        where_zero = np.where(data==0)[0]
                        if len(where_zero)>0:
                            info_str_15 = "\t -- {0} zeros in '{1}'. Skipping p_change...".format(len(where_zero),_key)
                            self.__vb_helper(verbose, info_str_15)
                            pass
                        else:
                            # Take percent difference
                            avg_p_change = calc_avg_p_change( data, where_nearest_neighbors )
                            if avg_p_change is None:
                                self.__vb_helper(verbose, "None in avg_p_change!? Should not happen...")
                            else:
                                # update into regr_dfs_per_class - all data available for regression
                                my_kwargs = dict()
                                for i in range(np.shape(avg_p_change)[0]):
                                    new_col_str = "APC{0}_{1}".format(self._n_neighbors_[i], _key)
                                    self.__vb_helper(verbose, "\t"+new_col_str)
                                    my_kwargs[new_col_str] = avg_p_change[i]
                                self._regr_dfs_per_class_[key] = self._regr_dfs_per_class_[key].assign(**my_kwargs)

                        self._avg_dist_dfs_per_class_[key][_key] = avg_dist

                else:
                    self.__vb_helper(verbose, "No regression data in '{0}'.".format(key))
                    pass

        info_str_17 = "TableData Done in {0:.2f} seconds.".format(time.time()-start_time)
        self.__vb_helper(verbose, info_str_17)


    def get_data(self, what_data = "full", return_df = False):
        """Get all data contained in DataTable object after omission and
        subset cleaning. (Before data processing for class and regression.)

        Parameters
        ----------
        what_data: str, optional
            Default is 'full' with other options 'input', or 'output'.
        return_df: bool, optional
            If True, return a pandas DataFrame object.
            If False (default), return a numpy array.

        Returns
        -------
        data: ndarray or DataFrame
            An object containing all data from loaded files
            that has not already been removed in pre-processing.
        """
        if (what_data.lower()=="full"):
            data = self._full_data_
        elif (what_data.lower()=="input"):
            data = self._input_
        elif (what_data.lower()=="output"):
            data = self._output_
        else:
            raise ValueError("'{0}' not supported. Try 'full', 'input', or 'output'.".format(what_data))
        if return_df:
            return data
        else:
            return data.values


    def get_binary_mapping_per_class(self,):
        """Get binary mapping (0 or 1) of the class data for each unique
        classification. For each classification, a value of 1 is given
        if the class data matches that classification. If they do not
        match, then a value of 0 is given.

        Example:
        classifications -> A, B, C
        class data      -> [ A, B, B, A, B, C ]
        binary mapping  -> [[1, 0, 0, 1, 0, 0]   (for class A)
                            [0, 1, 1, 0, 1, 0]   (for class B)
                            [0, 0, 0, 0, 0, 1]]  (for class C)

        Returns
        -------
        binary_class_data: ndarray
            N by M array where N is the number of classes and M is the
            number of classifications in the data set.
            Order is determined by '_unique_class_keys_'.
        """
        cls = self._class_col_
        # create a different array for each class - one against all
        binary_class_data = []
        for i in range(self.num_classes):
            where_class_is = np.where( cls == self._unique_class_keys_[i], 1, 0 ) # 1 for True, 0 for False
            binary_class_data.append( np.concatenate(where_class_is, axis=None) )

        return np.array(binary_class_data)


    def get_all_class_data(self, ):
        """Get all data related to classification.

        Returns
        -------
        _class_col_: array
            Array with original classification data.
        _unique_class_keys_: array
            Unique classes found in the classification data.
        _class_col_to_ids_: array
            Array where the original classification data has been
            replaced with their respective class IDs (integers).
        _class_id_mapping_: dict
            Dictionary with the mapping between a classification
            from the original data set and its class ID.
        _binary_data_: ndarray
            Iterating over the unique classes, classification data
            is turned into 1 or 0 if it matches the given class.
            Also see 'get_binary_mapping_per_class()'.
        """
        out1 = self._class_col_   # simply the class column
        out2 = self._unique_class_keys_  # unique values in class column
        out3 = self._class_col_to_ids_  # class column but class strings turned into integers (class IDs)
        out4 = self._class_id_mapping_  # maps between a class ID and the original class string
        _binary_data_ = self.get_binary_mapping_per_class() # for all classes, 1 where that class, 0 else
        return out1, out2, out3, out4, _binary_data_


    def get_all_regr_data(self,):
        """Get all data related to regression.

        Since each class can have its own unique set of regression data,
        the data is sorted in dictionaries where the classification
        is the key to that set of regression data.

        Returns
        -------
        _regr_inputs_: dict
            For each class, the input data with no cleaning.
        _regr_outputs_: dict
            For each class, the output data with no cleaning.
        _regr_dfs_per_class_: dict
            For each class, cleaned output data.
            Cleaned meaning only values that can be converted
            to floats and that are not Nan.
        """
        out1 = self._regr_inputs_
        out2 = self._regr_outputs_
        out3 = self._regr_dfs_per_class_
        return out1, out2, out3


    def info(self):
        """Print info for the TableData object. For descriptions see 'get_info()'."""
        print( "File List: \n{0}".format(np.array(self._files_)) )
        print( "df Index Keys: \n{0}\n".format(np.array(self._df_index_keys_)) )
        print("---- VERBOSE OUTPUT ----")
        print( *self._for_info_, sep = '\n' )


    def get_info(self):
        """Returns what info is printed in the 'info()' method.

        Returns
        -------
        _files_: list
            File paths where data was loaded from.
        _df_index_keys_: list
            Index keys added to the DataFrame object once
            multiple files are joined together such that one can
            access data by file after they were joined.
        _for_info_: list
            Running list of print statements that include but
            are not limited to what is shown if 'verbose=True'.
        """
        out1 = self._files_
        out2 = self._df_index_keys_
        out3 = self._for_info_
        return out1, out2, out3


    # # *****************
    # def get_full_data(self, return_df = False):
    #     """Get all data contained in DataTable object.
    #
    #     ! DEPRICATED - use get_data(what_data='full')
    #
    #     Parameters
    #     ----------
    #     return_df: bool, optional, default = False
    #         If True, return a pandas DataFrame object.
    #         If False, return a numpy array.
    #
    #     Returns
    #     -------
    #     full_data: ndarray
    #         An array containing all data from loaded files
    #         that has not been already removed in pre-processing.
    #     """
    #     if return_df:
    #         return self._full_data_
    #     else:
    #         return self._full_data_.values
    # # *****************
    #
    # # *****************
    # #def get_input_data(self,return_df=False):
    #     """Get all input data.
    #
    #     ! DEPRICATED - use get_data(what_data='input')
    #
    #     Parameters
    #     ----------
    #     return_df: bool, optional
    #         If True, return a pandas DataFrame object.
    #         If False (default), return a numpy array.
    #
    #     Returns
    #     -------
    #     input_data: ndarray
    #         An array containing all input data.
    #     """
    #     if return_df:
    #         return self._input_
    #     else:
    #         return self._input_.values
    # # *****************
    #
    # # *****************
    # def get_output_data(self,return_df=False):
    #     """! DEPRICATED - use get_data(what_data='output')"""
    #     if return_df:
    #         return self._output_
    #     else:
    #         return self._output_.values
    # # *****************
    #
    # # *****************
    # def get_classes_to_ids(self,):
    #     """DEPRICATED - use get_class_data -> _class_col_to_ids_"""
    #     return None
    # # *****************
    #
    # # *****************
    # def get_class_names(self,):
    #     """ ! DEPRICATED - use get_class_data -> _unique_class_keys_ """
    #     return None
    # # *****************
    #
    # # *****************
    # def get_regr_input_data(self,):
    #     """DEPRICATED - use get_all_regr_data -> _regr_inputs_
    #     Return: dict """
    #     return self.regr_inputs_
    # # *****************
    #
    # # *****************
    # def get_regr_output_data(self,):
    #     """DEPRICATED - use get_all_regr_data -> _regr_outputs_
    #     Return: dict """
    #     return self.regr_outputs_
    # # *****************
    #
    # # *****************
    # def get_regr_sorted_output_data(self,):
    #     """DEPRICATED - use get_all_regr_data -> _regr_dfs_per_class_
    #     Return: dict """
    #     return self._regr_dfs_per_class_
    # # *****************


    def plot_3D_class_data(self, axes=None, fig_size = (4,5), mark_size = 12, \
                            which_val = 0, save_fig=False, plt_str='0', \
                            color_list=None):
        """Plot the 3D classification data in a 2D plot.
        3 input axis with classification output.

        Parameters
        ----------
        axes: list, optional
            By default it will order the axes as [x,y,z] in the original order
            the input axis were read in. To change the ordering, pass a list
            with the column names.
            Example:
            The default orderd is col_1, col_2, col_3.
            To change the horizontal axis from col_1 to col_2 you would use:
                'axes = ["col_2", "col_1", "col_3"]'
        fig_size: tuple, optional, default = (4,5)
            Size of the figure. (Matplotlib figure kwarg 'fig_size')
        mark_size: float, optional, default = 12
            Size of the scatter plot markers. (Matplotlib scatter kwarg 's')
        which_val: int, default = 0
            Integer choosing what unique value to 'slice' on in the
            3D data such that it can be plotted on 2D.
            (If you had x,y,z data you need to choose a z value)
        save_fig: bool, default = False
            Save the figure in the local directory.
        plt_str: str, default = '0'
            If you are saving multiple figures you can pass a string
            which will be added to the end of the default:
            "data_plot_{plt_str}.pdf"
        color_list: list, default = None
        """
        full_data = self._full_data_
        input_data = self._input_

        if len(input_data) == 2:
            print("2D input data")
        if len(input_data) == 3:
            print("3D input data")

        if axes is not None:
            first_axis, second_axis, third_axis = axes
        else:
            first_axis, second_axis, third_axis = input_data.keys()
        print("Axes: {0}, {1}, {2}".format(first_axis, second_axis, third_axis) )

        if color_list:
            colors = color_list
            print("To set default colors for this object,\
                    re-istantiate using the option 'my_colors'.")
        else:
            colors = self._class_colors_

        color_dict = dict()
        for j, color_str in enumerate(colors):
            color_dict[j] = color_str


        ## MAKE LEGEND
        legend_elements = []
        for i, name in enumerate(self._unique_class_keys_):
            legend_elements.append( Line2D([], [], marker='s', color=color_dict[i],
                                           label=name, linestyle='None', markersize=8)  )
        #--------------

        # Specify value of other '3rd' axis not in 2D plot
        slice_value = np.unique(full_data[third_axis])[which_val]
        where_to_slice = full_data[third_axis] == slice_value  # boolean data frame

        # Find all indicies (rows) that have the slice value
        what_index = np.where( where_to_slice == True )[0]
        IDs = np.array( self._class_col_to_ids_ )
        data_to_plot = full_data[where_to_slice]

        # class IDs (ints 0->n) to color code
        class_to_colors = [ color_dict[val] for val in IDs[what_index] ]

        fig = plt.figure( figsize=fig_size, dpi=120 )
        plt.title( third_axis + '= %f'%(slice_value) )
        plt.scatter( data_to_plot[first_axis], data_to_plot[second_axis], \
                     c = class_to_colors, \
                     cmap = None, s=mark_size, marker = 's')

        plt.xlabel(first_axis); plt.ylabel(second_axis)
        plt.legend( handles = legend_elements, bbox_to_anchor = (1.03, 1.02))
        if save_fig: plt.savefig( "data_plot_{0}.pdf".format(plt_str), bbox_inches='tight')
        return fig

        
    def add_class_data_plot(self, fig, ax, which_ax=(0)):
        ax[which_ax].plot( np.random.rand(10) )
        return fig, ax
