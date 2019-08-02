import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

class sampler():

    def __init__( self, classifier ):
        self.classifier_obj = classifier
        #self.regressor_obj = regressor

        # Find the bounds of the walker
        self.max_vals = []
        self.min_vals = []
        input_data_cols = self.classifier_obj.table_data.get_input_data().T
        for data in input_data_cols:
            self.max_vals.append(max(data))
            self.min_vals.append(min(data))

        # You can save chains_history in here
        self._chain_step_hist_holder_ = dict()

    def analytic_target_dist( self, name, args ):
        mu, nu = args
        arg1 = - mu**2 - ( 9 + 4*mu**2 +8*nu )**2
        arg2 = - 8*mu**2 - 8*( nu - 2 )**2
        return (16)/(3*np.pi) * ( np.exp(arg1) + 0.5*np.exp(arg2) )

    def classifier_target_dist( self, classifier_name, args  ):
        correct_shape_args = np.array( [args] )
        max_probs, nan_locs = self.classifier_obj.return_probs( classifier_name, correct_shape_args, all_probs=False )
        # The second return of return_probs is a list filled with nan_locations
        return max_probs


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
                    N_draws_per_swap = 3, c_spacing = 1.2, alpha = 1, \
                    upper_limit_reject = 1e5, verbose = False ):
        """Runs a Paralel Tempered MCMC.

        >>>

        Parameters
        ----------
        T_max : float
            Sets the maximum temperature MCMC in the chain.
        N_tot : int
            The total number of iterations for the PTMCMC.
        target_dist : method with the form  F( name, element of step_history )
            The target distribution to sample.
            The 'name' argument specifies the classifier interpolator to use.
            (A 2D analytic function is provided - analytic_target_dist)
        N_draws_per_swap : int, default 3
            Number of draws to perform for each MCMC before swap proposals.

        Returns
        -------
        chain_step_history : dict
            Holds the step history for every chain. Keys are integers that range
            from 0 (max T) to the total number of chains -1 (min T).
        T_list : array
            Array filled with the temperatures of each chain from max to min.
         """
        # Create list of Temperatures - T_i+1 = T_i ^c
        # T_max -> T=1
        T_list = [T_max]
        while T_list[-1] > 1.3:
            T_list.append( T_list[-1]**(1/c_spacing) )
        T_list.append(1)
        T_list = np.array(T_list)

        num_chains = len(T_list)
        if verbose:
            print("Num chains: %i"%num_chains)
            print("Temperatures: \n", T_list )

        # plotting
        chain_holder = dict()
        for i in range( len(T_list) ):
            chain_holder[i] = []

        N_loops = int( N_tot/N_draws_per_swap )

        # Initial conditions for all chains
        this_iter_step_loc = [init_pos]*num_chains

        # Accept ratio tracker
        total_acc = np.zeros( num_chains )
        total_rej = np.zeros( num_chains )
        acc_ratio_holder = np.zeros( num_chains )

        for k in range(N_loops):
            # Number of draws before swap
            N_draws = N_draws_per_swap

            last_step_holder = []
            for i in range( num_chains ):
                # Run MCMC as f(T) N_draw times
                step_history= [this_iter_step_loc[i]]
                steps, acc, rej = self.run_MCMC( N_draws,
                                                 alpha, \
                                                 step_history, \
                                                 target_dist,\
                                                 classifier_name,\
                                                 T = T_list[i], \
                                                 upper_limit_reject = upper_limit_reject)
                last_step_holder.append( steps[-1] ) # save 'current' params for each T
                total_acc[i] += acc
                total_rej[i] += rej
                acc_ratio_holder[i] = total_acc[i]/(total_acc[i] + total_rej[i])

                # plotting data
                chain_holder[i].append( np.array(steps)  )

            if verbose:
                #print( "acc/total: {0}".format(acc_ratio_holder[0]), end="\r" )
                b = "num_acc/total: Tmax {0:.4}, Tmin {1:.4}".format( acc_ratio_holder[0], acc_ratio_holder[-1] )
                sys.stdout.write('\r'+b)

            # Calc H to see if chains SWAP
            accept = 0; reject = 0
            for i in range( len(T_list)-1 ):
                args_i = last_step_holder[i]
                args_i_1 = last_step_holder[i+1]

                top = (target_dist( classifier_name, args_i_1 ))**(1/T_list[i]) * \
                      (target_dist( classifier_name, args_i ))**(1/T_list[i+1])
                bot = (target_dist( classifier_name, args_i ))**(1/T_list[i]) * \
                      (target_dist( classifier_name, args_i_1 ))**(1/T_list[i+1])
                # inter-chain transition probability
                H = min( 1 , top/bot )

                chance = np.random.uniform(low=0, high=1)
                if chance <= H:
                    accept+=1
                    # SWAP the params between the two chains
                    last_step_holder[i] = args_i_1
                    last_step_holder[i+1] = args_i
                else:
                    reject+=1
            #print(accept, reject); print(last_step_holder)

            # Update current params (could be swapped and such)
            # to read into MCMC on next iteration!!!
            this_iter_step_loc = last_step_holder

        chain_step_history = dict()
        for pos, steps in chain_holder.items():
            chain_step_history[pos] = np.concatenate(steps)

        if verbose:
            print( "\nLength of chains: \n", np.array([len(chain_step_history[i]) for i in range(num_chains)]) )
            self.make_trace_plot( chain_step_history, T_list, 0, save_fig=False )
            self.make_trace_plot( chain_step_history, T_list, num_chains-1, save_fig=False )

        return chain_step_history, T_list


    def run_MCMC( self, N_trials, alpha, step_history, target_dist, \
                    classifier_name, T = 1, upper_limit_reject = 1e5):
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

        accept = 0; reject = 0;

        while (accept+reject) < N_trials and reject < abs(int(upper_limit_reject)):
            current_step = step_history[-1]

            # f(θ)
            val = target_dist( classifier_name, current_step )

            # θ+Δθ
            trial_step = current_step + np.random.normal( 0, alpha, size=len(current_step) )

            # f(θ+Δθ)
            trial_val = target_dist( classifier_name, trial_step )

            accept_prob = min( 1, (trial_val/val)**(1/T) )

            chance = np.random.uniform(low=0, high=1)
            if chance <= accept_prob:
                accept += 1
                step_history.append( trial_step )
            else:
                reject += 1

        return np.array(step_history), accept, reject


    def make_trace_plot( self, chain_holder, T_list, Temp, save_fig=False ):
        which_temp = Temp

        n_axis = len(chain_holder[which_temp].T)
        steps = chain_holder[which_temp].T

        fig, axs = plt.subplots(nrows=n_axis, ncols=2,
                                       figsize=(10,6), dpi=100,
                                       gridspec_kw={'width_ratios': [1.8, 1]})
        axis_names = self.classifier_obj.table_data.get_input_data(return_df=True).keys()
        for num, ax in enumerate(axs):
            ax[0].plot(steps[num], '-', color = "C4")
            ax[0].set_title( "Input: %s"%(axis_names[num]) )
            ax[0].set_ylabel( "Axis %i"%(num) + " , T=%.2f"%(T_list[which_temp]), fontsize=12 )
            ax[0].set_xlabel("N steps", fontsize=12)

            ax[1].hist(steps[num], bins=int(len(steps[num])/20),
                 histtype='step', density=True, color = "C1")
            ax[1].set_xlabel( "Axis %i"%num, fontsize=12 )
            ax[1].set_ylabel( "PDF", fontsize=12 )

        fig.subplots_adjust(hspace=0.4)
        plt.show()



    def normalize_step(self, step):
        return step
