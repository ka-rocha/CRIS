import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time



class sampler():

    def target_dist( arg ):
        mu, nu = arg
        arg1 = - mu**2 - ( 9 + 4*mu**2 +8*nu )**2
        arg2 = - 8*mu**2 - 8*( nu - 2 )**2
        return (16)/(3*np.pi) * ( np.exp(arg1) + 0.5*np.exp(arg2) )


    def run_MCMC( N_steps, alpha, step_history, T = 1):

        accept = 0; reject = 0;

        while accept <= N_steps and reject < 1e5:
            current_step = step_history[-1]

            # f(θ)
            val = target_dist( current_step )

            # θ+Δθ
            trial_step = current_step + np.random.normal( 0, alpha, size=len(current_step) )

            # f(θ+Δθ)
            trial_val = target_dist( trial_step )

            accept_prob = min( 1, (trial_val/val)**(1/T) )

            chance = np.random.uniform(low=0, high=1)
            if chance <= accept_prob:
                accept += 1
                step_history.append( trial_step )
            else:
                reject += 1

        return np.array(step_history), accept, reject


    start_theta = [0,0]
    step_history = [start_theta]

    steps, acc, rej = run_MCMC( 500, 1, step_history, T = 1 )
    print(acc, rej)
