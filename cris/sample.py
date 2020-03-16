import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
from collections import OrderedDict

import scipy.stats
from scipy.spatial.distance import pdist

# extra: something saying one extra point is not near other currently running point
#        maybe also include check that it's not already in a grid cell that has a simulation

class Sampler():
    """
    Sampler
    =======
    """

    def __init__( self, classifier = None, regressor = None ):
        self._Classifier_ = classifier
        self._Regressor_ = regressor

        if self._Classifier_ is None:
            pass
        else:
            # Find the bounds of the walker - should be a TableData attribute
            self._max_vals_ = []
            self._min_vals_ = []
            input_data_cols = self._Classifier_._TableData_.get_data(what_data='input').T
            for data in input_data_cols:
                self._max_vals_.append(max(data))
                self._min_vals_.append(min(data))
            # self._max_vals_, self._min_vals_ = self._Classifier_.get_data_bounds()

        # You can save chains_history in here
        self._chain_step_hist_holder_ = OrderedDict()
        # Not fully implemented yet I'm pretty sure....
        self._MAX_APC_str_ = []

    def TD_2d_analytic(self, name, args, **kwargs):
        """Two dimensional analytic target distribution for testing.
        $\frac{16}{3\pi} \left( \exp\left[-\mu^2 - (9 + 4\mu^2 + 8\nu)^2\right]
        + \frac{1}{2} \exp\left[- 8 \mu^2 - 8 (\nu-2)^2\right] \right)$
        """
        mu, nu = args
        arg1 = - mu**2 - ( 9 + 4*mu**2 +8*nu )**2
        arg2 = - 8*mu**2 - 8*( nu - 2 )**2
        return (16)/(3*np.pi) * ( np.exp(arg1) + 0.5*np.exp(arg2) )

    def TD_classification(self, classifier_name, args, **kwargs):
        """Target distribution using classification.
        Namely: f(x) = 1 - max[P(class)]

        If classification probability is Nan: f(x) = 0
        """
        TD_verbose = kwargs.get("TD_verbose", False)
        TD_BETA = kwargs.get("TD_BETA", 1.0)

        normalized_probs, where_not_nan = self._Classifier_.return_probs( classifier_name, args, \
                                                              verbose=TD_verbose )
        max_probs = np.max(normalized_probs, axis=1)
        if TD_verbose:
            print("max_probs  |  len(where_not_nan) != len(max_probs)  |  args")
            print(max_probs, len(where_not_nan) != len(max_probs), "\t BETA={}\n".format(TD_BETA))

        if (len(where_not_nan) != len(max_probs)):
            return 0
        else:
            theoretical_max_TD_cls_term = 1 - 1/self._Classifier_._TableData_.num_classes
            return ( (1-max_probs) * 1/theoretical_max_TD_cls_term )**(TD_BETA)

    def TD_classification_regression(self, names, args, **kwargs):
        """Target distribution using both classification & regression.

        kwargs
        TAU : float, optional
            Relative weight of classification to regression term.
            By default it's set to 0.6.
        """
        normalized_probs, where_not_nan = self._Classifier_.return_probs( names[0], args, \
                                                              verbose=False )
        max_probs = np.max(normalized_probs, axis=1)
        pred_class_ids = np.argmax(normalized_probs, axis=1 )
        cls_key = [self._Classifier_.class_id_mapping[i] for i in pred_class_ids]

        theoretical_max_TD_cls_term = 1 - 1/self._Classifier_._TableData_.num_classes
        classification_term = (1 - max_probs) * 1/theoretical_max_TD_cls_term

        if (len(where_not_nan) != len(max_probs)):
            return 0
        else:
            if isinstance(self._Regressor_.regr_dfs_per_class[cls_key[0]], pd.DataFrame):
                max_APC, which_col_max = self._Regressor_.get_max_APC_val( names[1], cls_key[0], args )
                self._MAX_APC_str_.append(which_col_max)

                A1 = kwargs.pop("A1", 0.5)
                scaling_log_func = lambda A1, x : np.log10( A1 * np.abs(x) + 1 )
                A0 = 1/scaling_log_func( A1, self._Regressor_.abs_max_APC )
                regression_term = A0 * scaling_log_func( A1, max_APC )
            else:
                regression_term = 0

        TAU = kwargs.pop("TAU", 0.5)
        TD_BETA = kwargs.pop("TD_BETA", 1.0)
        if kwargs.pop("TD_verbose", False):
            print( "TAU: {0} | TD_BETA: {1}".format(TAU, TD_BETA) )
        return ( TAU * classification_term + (1 - TAU) * regression_term )**(TD_BETA)


    def save_chain_step_history(self, key, chain_step_history, overwrite=False):
        """Save PTMCMC output chain_step_history inside the sampler object."""

        assert (key not in self._chain_step_hist_holder_.keys() or overwrite), \
                "\nYou are about to overwrite an existing element in '{0}'\
                            \n\n\tUse the option 'overwrite=True' to reassign.".format(key)

        self._chain_step_hist_holder_[key] = chain_step_history
        print("Saved chain to '{0}'.".format(key))


    def get_saved_chain_step_history(self, key, return_all=False):
        if return_all:
            return self._chain_step_hist_holder_
        else:
            return self._chain_step_hist_holder_[key]


    def run_PTMCMC( self, T_max, N_tot, init_pos, target_dist, classifier_name, \
                    N_draws_per_swap=3, c_spacing=1.2, alpha=1.0, \
                    upper_limit_reject=1e5, verbose=False, trace_plots=False, **TD_kwargs ):
        """Runs a Paralel Tempered MCMC with a user specified target distribution.
        Calls the method run_MCMC.

        Parameters
        ----------
        T_max : float
            Sets the maximum temperature MCMC in the chain.
        N_tot : int
            The total number of iterations for the PTMCMC.
        init_pos : array
            Initial position of walkers in each axis.
        target_dist : callable,  f( name, element of step_history )
            The target distribution to sample.
            The 'name' argument specifies the classifier interpolator to use.
            (A 2D analytic function is provided - analytic_target_dist)
        classifier_name : str, list
            A single string or list of strings specifying the interpolator to use
            for classification or classification and regression respectively.
        N_draws_per_swap : int, optional
            Number of draws to perform for each MCMC before swap proposals. (default 3)
        c_spacing : float, optional
            Sets the spacing of temperatures in each chain. (default 1.2)
            T_{i+1} = T_{i}^{1/c}, range: [T_max , T=1]
        alpha : float, optional
            Sets the standard deviation of steps taken by the walkers.
        upper_limit_reject : float, optional
            Sets the upper limit of rejected points. (default 1e5)
        verbose : bool, optional
            Useful print statements during execution. (default False)

        Returns
        -------
        chain_step_history : dict
            Holds the step history for every chain. Keys are integers that range
            from 0 (max T) to the total number of chains -1 (min T).
        T_list : array
            Array filled with the temperatures of each chain from max to min.

        Notes
        -----
        There is a prior on the PTMCMC which is to not go outside the range of
        training data in each axis.
         """
        # create list of Temperatures: T_{i+1} = T_{i}^{1/c}, range: [T_max , T=1]
        T_list = [T_max]
        while (T_list[-1] > 1.3):
            T_list.append( T_list[-1]**(1/c_spacing) )
        T_list.append(1)
        T_list = np.array(T_list)

        num_chains = len(T_list)
        if verbose:
            print("Num chains: {0}\nTemperatures: {1}\n".format(num_chains, T_list) )

        # data storage
        chain_holder = OrderedDict()
        for i in range( len(T_list) ):
            chain_holder[i] = []

        N_loops = int( N_tot/N_draws_per_swap )

        # Initial conditions for all links in chain
        # ADD: init_pos can be a unitless position in the range of the axes of the data
        #       This change should also be applied to alpha - user just gives num(0,1]
        if not isinstance( init_pos, np.ndarray ):
            init_pos = np.array( init_pos )
        if init_pos.ndim > 1:
            raise ValueError("init_pos has {0} dimensions, must be one dimensional.".format(init_pos.ndim))
        this_iter_step_loc = [init_pos]*num_chains

        # Accept ratio tracker
        total_acc = np.zeros( num_chains )
        total_rej = np.zeros( num_chains )
        acc_ratio_holder = np.zeros( num_chains )

        start_time = time.time()
        for counter in range(N_loops):
            # Number of draws before swap
            N_draws = N_draws_per_swap

            last_step_holder = []
            for i in range( num_chains ):
                # Run MCMC as f(T) N_draw times
                step_history= [this_iter_step_loc[i]]
                steps, acc, rej = self.run_MCMC( N_draws, alpha, step_history, \
                                                 target_dist, classifier_name,\
                                                 T = T_list[i], \
                                                 upper_limit_reject = upper_limit_reject, \
                                                 **TD_kwargs)
                last_step_holder.append( steps[-1] ) # save 'current' params for each T
                total_acc[i] += acc
                total_rej[i] += rej
                acc_ratio_holder[i] = total_acc[i]/(total_acc[i] + total_rej[i])

                # data storage
                chain_holder[i].append( np.array(steps)  )

            if verbose:
                # useful output during the PTMCMC
                num_bars = 20
                how_close = ( int( (counter/(N_loops-1))*num_bars ) )
                progress_bar = "|" + how_close*"=" + ">" + abs(num_bars-how_close)*" " + "|" \
                                + "{0:.1f}%".format( counter/(N_loops-1) * 100)
                b = "num_acc/total: Tmax {0:.4f}, Tmin {1:.4f}, loop # {2}, {3}".format( \
                        acc_ratio_holder[0], acc_ratio_holder[-1], counter, progress_bar )
                sys.stdout.write('\r'+b)

            # Calc H to see if chains SWAP
            accept = 0; reject = 0
            for i in range( len(T_list)-1 ):
                args_i = last_step_holder[i]
                args_i_1 = last_step_holder[i+1]

                top = (target_dist( classifier_name, args_i_1, **TD_kwargs ))**(1/T_list[i]) * \
                      (target_dist( classifier_name, args_i, **TD_kwargs ))**(1/T_list[i+1])
                bot = (target_dist( classifier_name, args_i, **TD_kwargs ))**(1/T_list[i]) * \
                      (target_dist( classifier_name, args_i_1, **TD_kwargs ))**(1/T_list[i+1])

                # can get div 0 errors when using linear because of nans
                if (bot == 0):
                    ratio = 0
                else:
                    ratio = top/bot

                # inter-chain transition probability
                H = min( 1 , ratio )

                chance = np.random.uniform(low=0, high=1)
                if chance <= H:
                    accept+=1
                    # SWAP the params between the two chains
                    last_step_holder[i] = args_i_1
                    last_step_holder[i+1] = args_i
                else:
                    reject+=1
            #print(accept, reject); print(last_step_holder)

            # Update current params (could be swapped)
            # to read into MCMC on next iteration!!!
            this_iter_step_loc = last_step_holder

        chain_step_history = OrderedDict()
        for pos, steps in chain_holder.items():
            chain_step_history[pos] = np.concatenate(steps)

        if verbose:
            print( "\nLength of chains: \n{0}".format(np.array([len(chain_step_history[i]) for i in range(num_chains)]))  )
            fin_time_s = time.time()-start_time
            print( "Finished in {0:.2f} seconds, {1:.2f} minutes.".format(fin_time_s, fin_time_s/60) )
            if trace_plots:
                self.make_trace_plot( chain_step_history, T_list, 0, save_fig=False )
                self.make_trace_plot( chain_step_history, T_list, num_chains-1, save_fig=False )

        return chain_step_history, T_list


    def run_MCMC( self, N_trials, alpha, step_history, target_dist, \
                    classifier_name, T = 1, upper_limit_reject = 1e5, **TD_kwargs):
        """Runs a Markov chain Monte Carlo for N trials.

        Parameters
        ----------
        N_trials : int
            Number of proposals or trials to take before stopping.
            (Not necessarily the number of accepted steps.)
            The MCMC will stop after 1e5 rejections automatically.
        alpha : float
            Related to the step size of the MCMC walker.
            Defines the standard deviation of a zero mean normal
            from which the step is randomly drawn.
        step_history : list
            Initial starting location in parameter space.
            Could contain an arbitrary number of previous steps
            but a walker will start at the last step in the list.
        targe_dist : method, takes args ( element of step_history )
            The target distribution to sample.
            A 2D analytic function is provided - analytic_target_dist
        classifier_name : str
            Name of interpolation technique used for classification.
            Class probabilties are the target distribution for
            the trained classifier.
        T : float
            Temperature of the MCMC. A higher temperature
            flattens the target distribution allowing more exploration.
        upper_limit_reject : int, default 1e5
            Sets the maximum number of rejected steps before the MCMC
            stops walking. Avoiding a slowly converging walk with few
            accepted points.

        Returns
        -------
        step_history : array
            An array containing all accepted steps of the MCMC.
        accept : int
            Total number of accepted steps.
        reject : int
            Total number of rejected steps.
        """
        if not isinstance( step_history, np.ndarray ):
            step_history = np.array(step_history)
        if step_history.ndim == 1:
            step_history = np.array([step_history])
        # We will be appending to the list
        step_history = list(step_history)

        accept = 0; reject = 0;

        while (accept+reject) < N_trials and reject < abs(int(upper_limit_reject)):
            current_step = step_history[-1]

            # f(θ)
            val = target_dist( classifier_name, current_step, **TD_kwargs )
            # θ+Δθ
            trial_step = current_step + np.random.normal( 0, alpha, size=len(current_step) )
            # f(θ+Δθ)
            trial_val = target_dist( classifier_name, trial_step, **TD_kwargs )

            # check if the trial step is in the range of data
            for i, step_in_axis in enumerate(trial_step):
                if (step_in_axis <= self._max_vals_[i] and step_in_axis >= self._min_vals_[i]):
                    pass
                else:
                    trial_val = 0 # essential reject points outside of range

            if (val == 0): # avoid div 0 errors
                ratio = 0
            else:
                ratio = trial_val/val

            accept_prob = min( 1, (ratio)**(1/T) )

            chance = np.random.uniform(low=0, high=1)
            if chance <= accept_prob:
                accept += 1
                step_history.append( trial_step )
            else:
                reject += 1

        return np.array(step_history), accept, reject


    def normalize_step_history(self, step_history):
        """Take steps and normalize [0,1] according to max and min in each axis.
        The max and min are taken from the original data set from TalbeData."""
        normed_steps = np.copy(step_history)
        for j, steps_in_axis in enumerate(step_history.T):
            normed_steps.T[j] = (steps_in_axis - self._min_vals_[j])/(self._max_vals_[j]-self._min_vals_[j])
        return normed_steps


    def undo_normalize_step_history(self, normed_steps):
        """Takes normed steps from [0,1] and returns their value
        in the original range of the axes based off the range of training data."""
        mapped_steps = np.copy(normed_steps)
        for j, steps_in_axis in enumerate(normed_steps.T):
            mapped_steps.T[j] = steps_in_axis * (self._max_vals_[j]-self._min_vals_[j]) + self._min_vals_[j]
        return mapped_steps

    def do_simple_density_logic(self, step_history, N_points, Kappa, var_mult = None,
                                add_mvns_together = False, verbose = False):
        """Perform multivariate normal density logic on a given step history.
        This is a simplified version of the method 'do_density_logic'. It assumes
        that every accepted point will have the same exact MVN.

        Parameters
        ----------
        step_history : ndarray
        N_points : int
        Kappa : float
        var_mult : float, ndarray, optional
        add_mvns_together : bool, optional
        verbose : bool, optional

        Returns
        -------
        accepted_points : ndarray
        rejected_points : ndarray
        """
        n_dim = len( self._Classifier_.input_data[0] )
        # approximate scaling of the variance given a set of N points to propose
        # the filling factor Kappa is not generally known a priori
        if var_mult is None:
            var_mult = 1
        sigma = Kappa * 0.5 * (N_points)**(-1./n_dim) * var_mult
        Covariance = sigma * np.identity( n_dim )
        if verbose: print("Covariance: \n{0}".format(Covariance))

        single_MVN = scipy.stats.multivariate_normal( np.zeros(n_dim), Covariance )
        max_val = 1/np.sqrt( (2*np.pi)**(n_dim) * np.linalg.det(Covariance) )

        accepted_points = []; rejected_points = []
        for step in step_history:
            if len(accepted_points) < 1:
                accepted_points.append( step )
                continue

            # distance: far is this point from all the accepted_points
            dist_to_acc_pts =  step - np.array(accepted_points)
            pdf_val_at_point =  single_MVN.pdf( dist_to_acc_pts )
            if isinstance(pdf_val_at_point, float):
                pdf_val_at_point = [pdf_val_at_point]

            if add_mvns_together:
                pdf_val_at_point = [ np.sum( pdf_val_at_point ) ]
            random_chances = np.random.uniform(low=0, high=max_val, size=len(pdf_val_at_point) )
            chance_above_distr = random_chances > pdf_val_at_point

            if chance_above_distr.all():
                accepted_points.append( step )
            else:
                rejected_points.append( step )

        return np.array(accepted_points), np.array(rejected_points)


    def do_density_logic( self, step_history, N_points, Kappa,\
                            shuffle = False, norm_steps = False, var_mult = None, \
                            add_mvns_together = False, pre_acc_points = None, verbose = False ):
        """Do the density based off of the normal gaussian kernel on each point. This method
        automatically takes out the first 5% of steps of the mcmc so that the initial starting
        points are not chosen automatically (if you start in a non-ideal region). Wait for
        the burn in.

        Parameters
        ----------
        step_history : ndarray
        N_points : int
        Kappa : float
        shuffle : bool, optional
        norm_steps : bool, optional
        var_mult : float, optional
        add_mvns_together : bool, optional
        verbose : bool, optional

        Returns
        -------
        accepted_points : ndarray
        rejected_points : ndarray
        accepted_sigmas : ndarray
        """

        if shuffle: # shuffle the order of the steps
            if verbose: print("Shuffling steps....")
            np.random.shuffle(step_history) # returns none

        if norm_steps: # normalize steps
            if verbose: print( "Normalizing steps...." )
            step_history = self.normalize_step_history( step_history )

        # Set the default average length scale
        num_dim = len( self._Classifier_.input_data[0] )
        sigma = Kappa * 0.5 * (N_points)**(-1./num_dim)

        # We assume the covaraince is the identity - later we may pass the entire array
        # but for now we just assume you pass a variance multiplier (var_mult)
        if var_mult is None:
            var_mult = np.array([1]*num_dim)
        else:
            var_mult = np.array( var_mult )
            if var_mult.ndim != num_dim:
                raise ValueError( "var_mult must be the same dimensionality as input data." )

        if verbose:
            print( "Num dims: {0}".format(num_dim) )
            print( "length scale sigma: {0}".format(sigma) )
            print( "var_mult: {0}".format(var_mult) )
            print( "Kappa: {0}".format(Kappa) )

        ### -> Forcing a few key points to always be accepted, for example
        if pre_acc_points is None:
            accepted_points = []
        elif isinstance(acc_pts, np.ndarray):
            accepted_points = list(acc_pts)
        else:
            accepted_points = []

        accepted_points = []
        accepted_sigmas = []
        max_val_holder = []
        accepted_mvn_holder = []
        rejected_points = []

        skip_steps = int(len(step_history)*0.05)
        good_steps = step_history[skip_steps:] # skip first 5% of steps to get into a good region

        for i in range(len(good_steps)):
            proposed_step = good_steps[i]

            accept = False
            if len(accepted_points) == 0:
                accept = True
            else:
                # If you enter you must have accepted one point
                k = len(Sigma)
                max_val = 1/np.sqrt( (2*np.pi)**(k) * np.linalg.det(Sigma) )
                max_val_holder.append( max_val )
                rnd_chance = np.random.uniform(low=0, high=np.max(max_val_holder), size=1)
                # we will choose the chance from [0, highest point in distr]

                distr_holder = []
                for point in accepted_mvn_holder:
                    eval_mvn_at_new_step = point.pdf( proposed_step )
                    distr_holder.append( eval_mvn_at_new_step )


                if add_mvns_together:
                    # instead of checking each individual point keeping all mvns seperate, we want to add them
                    # together and get upper bound
                    # IF we do this we need to change the MVN to not be normalized !!!!
                    # THE UNORMALIZED MVN IS NOT IMPLEMENTED
                    total_chance_above_distr = rnd_chance > np.sum(distr_holder)
                else:
                    total_chance_above_distr = np.sum( rnd_chance > distr_holder )


                if len(accepted_points) == total_chance_above_distr:
                    accept = True
                else:
                    pass   # REJECT

            if accept:
                # https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
                #corner = np.random.normal(0,0.1, 1)
                #A = np.array( [ [sigma, corner], [corner, sigma] ] )
                #Sigma = np.dot(A,A.transpose())
                #Sigma = [ [sigma*var_mult[0], 0.], [0., sigma*var_mult[1]] ]
                Sigma = sigma * np.identity( len(var_mult) ) * np.array( [var_mult] )
                mvn = scipy.stats.multivariate_normal( proposed_step, Sigma )

                accepted_mvn_holder.append( mvn )
                accepted_sigmas.append( Sigma )
                accepted_points.append( proposed_step )
            else:
                rejected_points.append( proposed_step )

        if verbose:
            print("Num accepted: {0}".format(len(accepted_points)) )
            print("Num rejected: {0}".format(len(rejected_points)) )

        if norm_steps:
            if verbose: print("Unormalizing steps....")
            accepted_points = self.undo_normalize_step_history(accepted_points)
            rejected_points = self.undo_normalize_step_history(rejected_points)

        return np.array(accepted_points), np.array(rejected_points), np.array(accepted_sigmas)


    def get_proposed_points( self, step_history, N_points, Kappa, \
                             shuffle=False, norm_steps=False, \
                             add_mvns_together=False, \
                             var_mult=None, seed=None, n_repeats=1,
                             max_iters = 1e4, verbose=False, **kwargs ):
        """Given a step history of an MCMC, get N proposed points spread out
        in parameter space.

        The desnity logic is not deterministic, so multiple iterations
        may be needed to converge on a desired number of proposed points.
        This method performs multiple calls to do_density_logic while
        changing Kappa in order to return the desired number of points. After
        n_iters instances of the correct number of N_points, the distibution
        with the largest average distance is chosen.

        Parameters
        ----------
        step_history : ndarray
        N_points : int
        Kappa : float
        shuffle : bool, optional
        norm_steps : bool, optional
        add_mvns_together : bool, optional
        var_mult : ndarray, optional
        seed : float, optional
        n_iters: int, optional
        verbose : bool, optional

        Returns
        -------
        acc_pts : ndarray
            Array of proposed points to be used as initial
            conditions in new simulations.
        Kappa : float
            Scaling factor which reproduced the desired number
            of accepted points.

        Notes
        -----
        Will automatically exit if it goes through 500 iterations
        without converging on the desired number of points.
        """

        if seed is not None:
            numpy.random.seed(seed = seed)
            print("Setting seed: {0}".format(seed))

        if verbose:
            print("Converging to {0} points, {1} times.".format(N_points, int(n_repeats)))

        enough_good_pts = False; how_many_good_pts = int(n_repeats)
        good_n_points = []; avg_distances = []; good_kappas = []
        iters = 1
        start_time = time.time()
        while ( not(enough_good_pts) and iters < max_iters):

            acc_pts, rej_pts = self.do_simple_density_logic( step_history, N_points, Kappa, \
                                                        var_mult = var_mult, \
                                                        add_mvns_together = add_mvns_together, \
                                                        verbose = False )

            average_dist_between_acc_points = np.mean( pdist( acc_pts ) )
            if len(acc_pts) == N_points:
                good_n_points.append( acc_pts )
                avg_distances.append( average_dist_between_acc_points )
                good_kappas.append( Kappa )
                if len( good_n_points ) >= how_many_good_pts:
                    enough_good_pts = True

            if verbose:
                if len(acc_pts) == N_points:
                    print_str = "  *{0}*  {1:2.2f}s".format( len(good_n_points), abs(start_time-time.time()) )
                    ending = "\n"
                else:
                    print_str = "" ; ending = "\r"
                print( "\t acc_pts: {0}, Kappa = {1:.3f}".format(len(acc_pts), Kappa) + print_str, end=ending )

            diff = abs( len(acc_pts) - N_points )

            change_factor = 0.01/(  max(1, np.log10(iters))  )
            if len(acc_pts) > N_points:
                Kappa = Kappa * (1 + change_factor*diff) # increase kappa
            elif len(acc_pts) < N_points:
                if (1 - change_factor*diff) < 0:
                    Kappa = Kappa * 0.1
                else:
                    Kappa = Kappa * (1 - change_factor*diff) # decrease kappa

            iters += 1

        if (iters == max_iters): print("Reached max iters before converging!")
        if verbose:
            print("\nFinal Kappa = {0}\nConverged in {1} iters.".format(Kappa, iters))

        # we want 1/r dependance to penalize closely spaced points
        where_best_distribution = np.argmax( avg_distances )
        best_acc_pts = good_n_points[where_best_distribution]
        best_Kappa = good_kappas[where_best_distribution]
        if verbose:
            print( "Average Distances: \n{0}".format( np.array(avg_distances)) )
            print( "Kappas: \n{0}".format(np.array(good_kappas)) )
            print( "loc: {0}".format(where_best_distribution) )

            #self.make_prop_points_plots(step_history, best_acc_pts)
        return best_acc_pts, best_Kappa


    def make_prop_points_plots(self, step_hist, prop_points, axes = (0,1),
                                    show_fig = True, save_fig=False ):
        """Plot the proposed / accepted points over the step history."""
        axis1, axis2 = axes
        plt.figure(figsize=(4,4), dpi=90)
        plt.scatter( step_hist.T[axis1], step_hist.T[axis2], alpha=0.5, label="step history" )
        plt.scatter( prop_points.T[axis1], prop_points.T[axis2], color='pink', label="accepted points" )
        plt.xlim( self._min_vals_[axis1], self._max_vals_[axis1] )
        plt.ylim( self._min_vals_[axis2], self._max_vals_[axis2] )
        plt.legend(loc='best')
        plt.show()
        return None


    def make_trace_plot( self, chain_holder, T_list, Temp, save_fig=False, show_fig=True ):
        """Make a trace plot of the position of a sampler in an axis vs step number.
        This function makes titles assuming you are using the data from classifier."""

        if show_fig == False and save_fig == False:
            return

        which_temp = Temp # int? index

        n_axis = len(chain_holder[which_temp].T)
        steps = chain_holder[which_temp].T
        axis_names = self._Classifier_._TableData_.get_data(what_data='input',return_df=True).keys()

        fig, axs = plt.subplots(nrows=n_axis, ncols=2, figsize=(10, 2.5*n_axis), dpi=100, \
                                       gridspec_kw={'width_ratios':[1.8, 1]})

        for num, ax in enumerate(axs):
            ax[0].plot(steps[num], '-', linewidth=0.5,color = "C4")
            ax[0].set_title( "Input: {0}".format(axis_names[num]) )
            ax[0].set_ylabel( "Axis {0}".format(num) + " , T={0:.2f}".format(T_list[which_temp]), fontsize=12 )
            ax[0].set_xlabel( "N steps", fontsize=12)

            ax[1].hist(steps[num], bins= 50, \
                            histtype='step', density=True, color="C1")
            ax[1].set_xlabel( "Axis {0}".format(num), fontsize=12 )
            ax[1].set_ylabel( "Posterior", fontsize=12 )

        fig.subplots_adjust(hspace=0.45)
        if save_fig:
            plt.savefig( "trace_plot_T{0:.0f}.pdf".format(T_list[which_temp]) )
        if show_fig:
            plt.show()
        return None
