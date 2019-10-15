import numpy as np
import itertools
import pickle
import time
import matplotlib.pyplot as plt
from scipy.linalg import logm
import multiprocessing as mp
from sys import stdout


class DPH(object):

    def __init__(self, phases=None, observations=None, initial=None, transition=None, generator=None,
                 cores=mp.cpu_count()):
        if transition is not None:
            self.phases = len(transition)
        else:
            self.phases = phases
        self.observations = observations
        self.trace = {}
        self.cores = cores
        self.initial = initial
        self.transition = transition
        self.likelihood = 0
        self.estimate = None
        self.generator = generator

    def __str__(self):
        out = "Number of observations: %s\n" % len(self.observations)
        out += "Phases: %s\n" % self.phases
        out += "Method: %s\n\n" % self.estimate
        if self.initial is not None:
            out += "------------" * self.phases
            out += "\n"
            out += "Initial state vector\n"
            for a in range(self.phases):
                out += "% 9.8f " % self.initial[a]
            out += "\n\n"
            out += "------------" * self.phases
            out += "\n"
            out += "Transitions\n"
            for a in range(self.phases):
                for b in range(self.phases):
                    out += " %9.8f " % self.transition[a, b]
                out += "\n"
            out += "\n"
        if self.generator is not None:
            out += "------------" * self.phases
            out += "\n"
            out += "Generator\n"
            for a in range(self.phases):
                for b in range(self.phases):
                    if self.generator[a, b] >= 0:
                        out += " %9.8f " % self.generator[a, b]
                    else:
                        out += "%9.8f " % self.generator[a, b]
                out += "\n"
        return out

    def acf_plot(self, chain, lags, thin=1, save_as=None, fmt='eps'):
        trace = self.trace['parameters'][chain]
        freeparams = self.free_parameters()
        for i, parameter in enumerate(freeparams):
            series = [a[i] for a in trace]
            series += -np.mean(series)
            series = series[::thin]
            plt.acorr(series, maxlags=lags, color='b')
            plt.title("Autocorrelation Plot for free parameter %s" % parameter)
            plt.xlabel("Lag")
            plt.ylabel("ACF")
            if save_as is not None:
                plt.savefig(save_as + str(i) + '.' + fmt, format=fmt, dpi=1000)
            plt.show()

    def trace_plot(self, thin=1, save_as=None, fmt='eps'):
        """
        Prepares a trace plot of the log likelihood of the parameter stream in the trace.
        :param thin:    If supplied this is the number of observations to skip between samples
        :param save_as: If supplied this should be a file name without the extension to save the trace as.
        :return: A plot shown on screen or a plot saved in .eps format.
        """
        freeparams = self.free_parameters()
        for i, parameter in enumerate(freeparams):
            for chain in range(self.trace['chains']):
                trace = self.trace['parameters'][chain]
                series = [a[i] for a in trace]
                plt.plot(series[:: thin])
            plt.title("Parameter trace for free parameter %s" % parameter)
            plt.ylabel("Value")
            plt.xlabel("Sample")
            if save_as is not None:
                filename = save_as
                # if self.trace['chains'] > 1:
                #     filename += str(trace)
                plt.savefig(filename + '.' + fmt, format=fmt, dpi=1000)
                # trace += 1
            plt.show()

    def gr_plot(self, end, save_as=None, fmt='eps'):
        """
        Produce Gelman-Rubin plot to determine appropriate burn-in for an mcmc chain.
        :param end:     When to stop shortening the window.
        :param save_as: Saves the image as an eps file with the supplied file name.
        :return:
        """

        max_gr = []
        ptuples = list(zip(*[list(zip(*self.trace['parameters'][a])) for a in range(self.trace['chains'])]))
        for window in range(2, end):
            gr_stat = []
            for param in ptuples:
                test = [list(a[0: window]) for a in param]
                gr_stat += [gelman_rubin(test)]
            max_gr += [max(gr_stat)]
        plt.plot(max_gr)
        plt.ylim((0, max(max_gr) + 1))
        plt.title("Maximum Gelman-Rubin Statistic across all free parameters.")
        plt.xlabel("Window starts at iteration")
        plt.ylabel("Max Gelman-Rubin")
        if save_as is not None:
            plt.savefig(save_as + '.' + fmt, format=fmt, dpi=1000)
        plt.show()

    def parameters(self, start, end, thin=1):
        """
        Computes parameter estimates for MCMC methods by averaging the trace.  The resulting averages are then used to
        compute the log likelihood. CAUTION: This is probably not appropriate unless the GR plots show convergence.
        :param start: Where to start averaging from
        :param end: Where to end the averaging
        :param thin: The extent to which the trace should be thinned.
        :return: Updates the estimate of the initial probabilities, transition matrix and the log likelihood.
        """

        initial_mean = np.zeros(self.phases)
        transition_mean = np.zeros((self.phases, self.phases + 1))
        divisor = 2 * len(list(range(start, end, thin)))
        for chain in range(self.trace['chains']):
            initial_mean += sum([self.trace[chain]['initials'][i] for i in range(start, end, thin)])
            transition_mean += sum([self.trace[chain]['transitions'][i] for i in range(start, end, thin)])
        initial_mean /= divisor
        transition_mean /= divisor
        self.likelihood = sum([np.log(initial_mean @ np.linalg.matrix_power(transition_mean[:, : self.phases], obs - 1)
                                      @ transition_mean[:, self.phases]) for obs in self.observations])
        self.initial = initial_mean
        self.transition = transition_mean[:, : self.phases]

    def dens_plot(self, save_as=None, fmt='eps'):
        """
        Create a density plot using the parameter estimates.
        :param save_as: A filename to save the plot image as.
        :return: A plot of the density with the normalised data.
        """
        bins = range(min(self.observations), max(self.observations) + 1)
        obs = self.observations.copy()
        dens = []
        trans = self.transition
        absorb = 1 - trans.sum(axis=1)
        initial = self.initial
        for h in bins:
            dens += [initial @ np.linalg.matrix_power(trans, h - 1) @ absorb]
        dens = [dens[0]] + dens
        plt.figure(figsize=(12, 8))
        plt.step([0] + list(bins), dens, color='red', label='Estimated')
        obs = [obs.count(i) / len(obs) for i in bins]
        obs = [obs[0]] + obs
        plt.step([0] + list(bins), obs, color='darkgray', label='Observed')
        plt.title("Fit to observed data - %s phases" % self.phases, fontsize=24)
        plt.ylabel("Density", fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Stopping time", fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(loc='best', fontsize=18)
        if save_as is not None:
            plt.savefig(save_as + '.' + fmt, format=fmt, dpi=1000)
        plt.show()

    def fit_plot(self, start, end, thin=1, save_as=None, fmt='eps', transparency=0.02):
        """
        Plot a normed histogram of the observed data together with pdf estimates from the trace.  This will also
        populate self.initial, self.transition and self.likelihood based on the chosen parameters.
        :param start:           Where in the trace should the pdf calculations commence.  This should be at a point
                                after convergence in log likelihood
        :param end:             Where in the trace should the calculations cease
        :param thin:            Gibbs sampling produces serially correlated estimates.  This is addressed by taking
                                every kth parameter estimate
        :param save_as:         Putting a path/filename here will save plot as an .eps image.
        :param fmt:             The graphics format to use when saving
        :param transparency:    The transparency of the steps in the plot.
        :return:                A plot
        """
        bins = range(min(self.observations), max(self.observations) + 1)
        obs = self.observations.copy()
        self.parameters(start, end, thin)
        plt.figure(figsize=(12, 8))
        for chain in range(self.trace['chains']):
            for sample in range(start, end, thin):
                init = self.trace[chain]['initials'][sample]
                trans = self.trace[chain]['transitions'][sample][:, : self.phases]
                absorb = self.trace[chain]['transitions'][sample][:, self.phases]
                dens = []
                for h in bins:
                    dens += [init @ np.linalg.matrix_power(trans, h - 1) @ absorb]
                dens = [dens[0]] + dens
                plt.step([0] + list(bins), dens, color='red', alpha=transparency)
        obs = [obs.count(i) / len(obs) for i in bins]
        obs = [obs[0]] + obs
        plt.step([0] + list(bins), obs, color='black', label='Observed')
        plt.title("Fit to observed data - %s phases" % self.phases, size=20)
        plt.ylabel("Density", size=16)
        plt.xlabel("Stopping time", size=16)
        if save_as is not None:
            plt.savefig(save_as + '.' + fmt, format=fmt, dpi=1000)
        plt.show()

    def pickle_trace(self, name):
        """
        Pickles the trace so that it can be recovered later.
        :param name: The file name for the pickled data
        """
        pickle.dump(self.trace, open(name, 'wb'))

    def unpickle_trace(self, name):
        """
        Does what it says on the can.
        :param name: The name of the file containing the trace to unpickle.
        """
        self.trace = pickle.load(open(name, 'rb'))
        self.estimate = self.trace['form']

    def gibbs_dph(self, chains, length, method='DPH', prior=None):
        """
        Performs gibbs sampling using the generic DPH form
        :param chains:  The number of chains to run.  Multiple chains are used to test for convergence.
        :param length:  The desired length of the Gibbs chain
        :param method:  The method to use for the estimate.  One of 'DPH'(default), 'ADPH' or 'CF1'
        :param prior:   A tuple of numpy arrays specifying the priors to use in the estimation. prior[0] should be the
                        vector of priors to use when estimating the initial vector and prior[1] should be the array of
                        priors to use for the transitions.
        :return:        Updates the trace
        """

        def enforce_cf1(iterable):
            """
            Needed to enforce adherence to the Cf1 form when generating stochastic sub-matrices.
            :param iterable: A numpy array of priors
            :return:
            """
            output = np.zeros((self.phases, self.phases + 1))
            ub = 1
            for loc, phase in enumerate(iterable):

                # Since the cf1 form limits transitions to remaining in a given state or transitioning to the next
                # higher state we have a beta distribution.
                prob = np.random.beta(phase[loc], phase[loc + 1])
                output[loc, loc] = min(prob, ub)
                output[loc, loc + 1] = 1 - output[loc, loc]
                ub = output[loc, loc]
            return output

        self.estimate = method
        free_parameters = self.free_parameters()

        # Set up prior
        if prior is None:
            initial_prior = np.ones(self.phases)

            # For the generic DPH form any transition is possible.
            if method == 'DPH':
                transition_prior = np.ones((self.phases, self.phases + 1))

            # For the ADPH form only transitions to the same or higher states are allowed.
            elif method == 'ADPH':
                transition_prior = np.zeros((self.phases, self.phases + 1))
                for i in range(self.phases):
                    transition_prior[i, i: self.phases + 1] += np.ones((self.phases + 1 - i))

            # For the CF! form only transition to the same or NEXT higher state are allowed.
            elif method == 'CF1':
                transition_prior = np.zeros((self.phases, self.phases + 1))
                transition_prior[:, : self.phases] += np.eye(self.phases)
                transition_prior[:, 1: self.phases + 1] += np.eye(self.phases)

            else:
                # Todo: put something meaningful here
                pass

        # If priors are supplied, use them
        else:
            initial_prior = prior[0]
            transition_prior = prior[1]

        # Initialise chains
        # self.trace['form'] = type
        self.trace['chains'] = chains
        self.trace['length'] = length

        # The parameter lists are used for computing the Gelman Rubin statistics.
        self.trace['parameters'] = [[] for i in range(chains)]

        for chain in range(chains):
            # Start the timer
            start_time = time.time()

            # Initalise the trace
            self.trace[chain] = {}
            initial_dict = {}
            transition_dict = {}
            ln_likelihood = {}

            # Initialisation draw using only the prior
            initial_states = np.random.dirichlet(initial_prior)

            if method == 'DPH':
                transitions = np.array([np.random.dirichlet(phase) for phase in transition_prior])

            elif method == 'ADPH':
                transitions = np.zeros((self.phases, self.phases + 1))
                for i in range(self.phases):
                    transitions[i, i: self.phases + 1] = np.random.dirichlet(transition_prior[i, i: self.phases + 1])
            else:
                transitions = enforce_cf1(transition_prior)

            for sample in range(length):

                pool = mp.Pool(self.cores)

                # Step 1 of the Gibbs chain: Obtain a list of initial state counts and state transition counts for each
                # observation conditional on the initial state probabilities and transition probabilities
                sample_paths = pool.starmap_async(sample_path_dph, [(initial_states, transitions[:, : -1],
                                                                     observation, self.phases, method) for observation
                                                                    in self.observations]).get()

                pool.close()

                starts = initial_prior + sum([start[0] for start in sample_paths])
                paths = transition_prior + sum([path[1] for path in sample_paths])

                # Step 2 of the Gibbs chain: Obtain initial state probabilities and transition probabilities conditional
                # on the initial state counts and state transition counts.
                initial_states = np.random.dirichlet(np.asarray(starts))
                plist = [initial_states[b] for b in [a[1] for a in free_parameters if a[0] == 'initials']]

                if method == 'DPH':
                    transitions = np.array([np.random.dirichlet(phase) for phase in paths])

                elif method == 'ADPH':
                    transitions = np.zeros((self.phases, self.phases + 1))
                    for i in range(self.phases):
                        transitions[i, i: self.phases + 1] = np.random.dirichlet(paths[i, i: self.phases + 1])

                elif method == 'CF1':
                    transitions = enforce_cf1(paths)

                plist += [transitions[b[0], b[1]] for b in [a[1] for a in free_parameters if a[0] == 'transitions']]

                initial_dict[sample] = initial_states.copy()
                transition_dict[sample] = transitions.copy()
                ln_likelihood[sample] = sum([ll[2] for ll in sample_paths])
                self.trace['parameters'][chain] += [plist]

                stdout.write("\rChain %s percent Compete: %3.2f" % (chain, 100 * (sample + 1) / length))
                stdout.flush()

            self.trace[chain]['initials'] = initial_dict
            self.trace[chain]['transitions'] = transition_dict
            self.trace[chain]['log likelihood'] = ln_likelihood

            # Stop the timer
            end_time = time.time()
            print("\nFinished chain %s  --- duration %5.4f seconds" % (chain, end_time - start_time))
            self.trace['form'] = method
        print("Finished!")

    def free_parameters(self):
        """
        Create a list of the free parameters.
        :return: A list of lists.
        """
        parameters = []
        for i in range(self.phases - 1):
            parameters += [['initials', i]]
        if self.estimate == 'DPH':
            for i in range(self.phases):
                for j in range(self.phases):
                    parameters += [['transitions', (i, j)]]
        elif self.estimate == 'ADPH':
            for i in range(self.phases):
                for j in range(i, self.phases):
                    parameters += [['transitions', (i, j)]]
        elif self.estimate == 'CF1':
            for i in range(self.phases):
                parameters += [['transitions', (i, i)]]
        return parameters

    def em_dph(self, tolerance, epochs, method='DPH', randomize=False):

        """
        Uses expectation maximisation to compute the parameters of a dph.  The update process will continue until the
        improvement falls below a user defined tolerance or a user defined number of iterations is complete.
        :param tolerance:   The smallest acceptable improvement at which the process will stop
        :param epochs:      The number of iterations before the process stops
        :param method:      Indicates the form of the transition matrix to estimate.  'DPH' (default), 'ADPH' or 'CF1'
        :param randomize:   Boolean indicating if the starting parameters should be randomised.
        :return:            Updates the initial and transition parameters
        """

        # Create a dictionary of observation counts.
        obs_dict = {a: self.observations.count(a) for a in set(self.observations)}

        # Create initial guess
        init_old = np.array([1 / self.phases] * self.phases)

        trans_old = np.zeros((self.phases, self.phases + 1))
        if method == 'ADPH':
            for i in range(self.phases):
                if randomize:
                    trans_old[i, i: self.phases + 1] = np.random.dirichlet([1] * (self.phases + 1 - i))
                else:
                    trans_old[i, i: self.phases + 1] = np.array([1 / (self.phases + 1 - i)] * (self.phases + 1 - i))
            self.estimate = 'EMadph'
        elif method == 'DPH':
            if randomize:
                trans_old = np.array([np.random.dirichlet([1] * (self.phases+ 1)) for i in range(self.phases)])
            else:
                trans_old = np.array([[1 / (self.phases + 1)] * (self.phases + 1) for i in range(self.phases)])
            self.estimate = 'EMdph'
        elif method == 'CF1':
            ub = 1
            for i in range(self.phases):
                if randomize:
                    prob = min(np.random.beta(1, 1), ub)
                    trans_old[i, i] = prob
                    trans_old[i, i + 1] = 1 - prob
                    ub = prob
                else:
                    trans_old[i, i: i + 2] = np.array([0.5, 0.5])
            self.estimate = 'EMcf1'
        else:
            print("method must be one of 'DPH', ADPH' or 'CF1'")

        # Initialise control variables
        converged = False
        iteration = 0

        # Main loop
        while not converged:
            init_new = np.zeros(self.phases)
            trans = np.zeros((self.phases, self.phases + 1))

            # Iterate over the observed absorption times.
            for time in obs_dict.keys():

                # Compute the expected sample path.  This is the Expectation step.
                exp_params = exp_sample_path_dph(init_old, trans_old[:, : -1], time, self.phases, method)
                init_new += exp_params[0] * obs_dict[time]
                trans += exp_params[1] * obs_dict[time]

            # Compute the expected initial state vector.  This is the Maximisation step
            self.initial = init_new / init_new.sum()

            # Compute the expected transition probabilities.  This is the Maximisation step
            self.transition = trans / trans.sum(axis=1)[:, None]
            if method=='CF1':
                diags = np.diag(self.transition[:, :-1])
                for i in range(self.phases - 1):
                    if diags[i] < diags[i + 1]:
                        self.transition[i, i] = diags[i + 1]
                        self.transition[i, i + 1] = 1 - diags[i + 1]
            # Compute convergence statistic
            i_tol = ((self.initial - init_old) ** 2).sum()
            t_tol = ((self.transition - trans_old) ** 2).sum()
            improvement = np.sqrt(i_tol + t_tol)

            # Test for convergence or timeout.
            if improvement < tolerance or iteration >= epochs:
                converged = True
                print("\nFinished!")

                # Trim the transition matrix.
                self.transition = self.transition[:, :-1]

            else:
                init_old = self.initial.copy()
                trans_old = self.transition.copy()
                stdout.write("\rIteration: %s.  Improvement: %10.8f" % (iteration, improvement))
                stdout.flush()
                iteration += 1

    def reg_gen(self, horizon):
        """
        Finds a generator of a transition matrix using Quasi-optimisation of the generator as described in Kreinin and
        Sidelnikova 'Regularization Alogorithms for Transition matrices' 2019
        :param horizon: The time period over which the transition matrix applies.
        :return:        A sub-stochastic generator matrix.
        """

        # The routine only works if there is a transition matrix
        if self.transition is not None:

            # The matrix needs to be a full matrix so the transitions to the absorbing state are included.
            b = (1 - self.transition.sum(axis=1)).reshape((self.phases,1))
            mat = np.hstack((self.transition, b))
            mat = np.vstack((mat, np.array([0] * self.phases + [1])))

            # The first estimate of the generator is the matrix logarithm divided by the horizon
            gen_est = (logm(mat) / horizon)[: self.phases]

            # The rows of the matrix logarithm are permuted into ascending order.
            row_order = np.argsort([a + a.mean() for a in gen_est])
            ordered_est = np.sort(gen_est, axis=1)

            # The problem is solved row by row.
            for i in range(self.phases):
                ell = 1
                while (self.phases - ell + 1) * ordered_est[i, ell + 1] - (ordered_est[i, 0] +
                                                                           ordered_est[i, ell + 1:].sum()) < 0:
                    ell += 1
                pivot = ell
                opt_est = np.zeros(self.phases + 1)
                for j in range(self.phases + 1):
                    if j not in range(1, pivot + 1):
                        opt_est[j] = ordered_est[i, j] - (ordered_est[i, 0] +
                                                          ordered_est[i, pivot + 1:].sum()) / (self.phases - pivot + 1)
                gen_est[i] = opt_est[np.argsort(row_order[i])]
            self.generator = gen_est[:, :-1]
        else:
            print("reg_gen needs a transition matrix to work.")


def gelman_rubin(chains):
    """
    Computes the Gelman-Rubin statistic to test for convergence of an MCMC process.
    :param chains: A numpy array containing at least two chains.
    :return: A scalar value representing the GR statistic
    """
    try:
        # Variable names mirror those in Bayesian Data Analysis 2nd Ed Gelman, Carlin, Stern and Rubin.
        m = len(chains)
        n = len(chains[0])
        phi_j = np.array([np.mean(chains[a]) for a in range(m)])
        phi = phi_j.mean()
        B = n / (m - 1) * np.square(phi_j - phi).sum()
        W = np.mean(np.array([np.square(np.std(chains[a], ddof=1)) for a in range(m)]))
        R_hat = (W * (n-1)/n + B / n)
        return np.sqrt(R_hat / W)
    except TypeError or ValueError:
        print("At least 2 chains are needed to calculate Gelman-Rubin statistic.")


def exp_sample_path_dph(initial_states, transitions, time_to_absorption, number_of_phases, method='DPH'):
    """
    Computes the expectation of time spent in each state or transitioning between states for a given time to absorption
    conditional on a given initial state vector and sub-stochastic transition matrix.
    :param initial_states:     A probability vector of initial states.
    :param transitions:        A sub-stochastic transition matrix
    :param time_to_absorption: An integer valued time to absorption
    :param number_of_phases:   The number of phases in the process.
    :param method:             One of 'DPH', 'ADPH', or 'CF1'
    :return:                   The expectation over the starting states and the expectation over the transitional
                               states.
    """

    # Initialise the return variables
    exp_trans = np.zeros((number_of_phases, number_of_phases + 1))
    eye = np.eye(number_of_phases)

    # The absorption probabilities
    b = 1 - transitions.sum(axis=1)

    # The probability of observing the given time to absorption
    lhood = initial_states @ np.linalg.matrix_power(transitions, (time_to_absorption - 1)) @ b

    exp_init = np.linalg.matrix_power(transitions, time_to_absorption - 1) @ b * initial_states / lhood
    for time in range(time_to_absorption - 1):

        # Pr(Y=y|X(k+1) = j
        cpabsgj = np.linalg.matrix_power(transitions, time_to_absorption - (time + 1) - 1) @ b

        # Pr(X(k+1)=j|X(k=i)
        cpjgi = initial_states @ np.linalg.matrix_power(transitions, time)

        # Take the outer product divide by the likelihood and multiply by the transition probabilities.
        exp_trans[:, : number_of_phases] += (np.outer(cpjgi, cpabsgj) / lhood) * transitions

    # expected last state prior to absorption.
    exp_last = (initial_states @ np.linalg.matrix_power(transitions, time_to_absorption - 1) / lhood) * b
    exp_trans[:, number_of_phases] += exp_last

    # Todo: eliminate these loops as they slow down the code.
    # for phase_i in range(number_of_phases):
        # exp_init[phase_i] = eye[phase_i] @ np.linalg.matrix_power(transitions, time_to_absorption - 1) @ b * initial_states[phase_i]
        # exp_init[phase_i] /= lhood
        # if time_to_absorption >= 2:
        #     if method != 'DPH':
        #         start = phase_i
        #     else:
        #         start = 0
        #     for phase_j in range(start, number_of_phases):
        #         for time in range(time_to_absorption - 1):
        #             prob = eye[phase_j] @ np.linalg.matrix_power(transitions, time_to_absorption - (time + 1) - 1) @ b
        #             prob *= initial_states @ np.linalg.matrix_power(transitions, time) @ eye[:, phase_i]
        #             prob /= lhood
        #             exp_trans[phase_i, phase_j] += prob * transitions[phase_i, phase_j]
        # prob = initial_states @ np.linalg.matrix_power(transitions, time_to_absorption - 1) @ eye[:, phase_i]
        # prob /= lhood
        # exp_trans[phase_i, number_of_phases] += prob * b[phase_i]
    return (exp_init, exp_trans)


def sample_path_dph(initial_states, transitions, time_to_absorption, number_of_phases, type):
    """
    Generates a sample path by reversing the Markov chain for discrete phase type distribution
    :param initial_states:      The initial state probabilities
    :param transitions:         A sub-stochastic transition matrix
    :param time_to_absorption:  The time at which the absorbing states was reached
    :param number_of_phases:    The number of phases in the Markov chain
    :return: A full path matrix of states visited before absorption.
    """
    path_matrix = np.zeros((number_of_phases, number_of_phases + 1))
    initial_phase = np.zeros(number_of_phases)

    eye = np.eye(number_of_phases)
    b = 1 - transitions.sum(axis=1)
    ln_likelihood = np.log(initial_states @ np.linalg.matrix_power(transitions, time_to_absorption - 1) @ b)

    if type == 'CF1':
        # In the cf1 form the process will be in the last phase prior to absorption.
        phase = number_of_phases - 1
        old_phase = phase
        path_matrix[phase, phase + 1] = 1

        # Iterate backwards from the absorbing state to the initial state.
        for time in range(time_to_absorption - 1, 0, -1):
            # Compute the probability of remaining in a given phase
            stay = (initial_states @ np.linalg.matrix_power(transitions, time - 1))[phase] * transitions[phase, phase]
            stay /= (initial_states @ np.linalg.matrix_power(transitions, time))[phase]

            # If a uniform random variable exceeds the probability of staying ina phase the process will move to an earlier
            # phase
            if np.random.rand() > stay:
                phase += -1
                ln_likelihood += np.log(1 - stay)
            else:
                ln_likelihood += np.log(stay)

            # Record the transition in the path array
            path_matrix[phase, old_phase] += 1
            old_phase = phase

        # Record the last state the process was in
        initial_phase[phase] = 1

    else:
        phase = number_of_phases
        old_phase = phase
        limit = number_of_phases

        # Iterate backwards from time of absorption to initial state.
        for time in range(time_to_absorption, 0, -1):
            probabilities = np.zeros(number_of_phases)
            for state in range(0, limit):

                # Probabilities just prior to absorption
                if time == time_to_absorption:
                    p_time = initial_states @ np.linalg.matrix_power(transitions, time - 1) @ eye[:, state] * b[state]
                    p_time /= initial_states @ np.linalg.matrix_power(transitions ,time - 1) @ b
                else:

                    # Probabilities for earlier times
                    p_time = initial_states @ np.linalg.matrix_power(transitions, time - 1) @ eye[:, state]
                    p_time *= eye[state, :] @ transitions @ eye[:, phase]
                    p_time /= initial_states @ np.linalg.matrix_power(transitions, time) @ eye[:, phase]

                probabilities[state] = p_time

            # Choose previous state and record it in the sample path matrix
            phase = np.random.choice(number_of_phases, 1, False, probabilities)[0]
            ln_likelihood += np.log(probabilities[phase])
            path_matrix[phase, old_phase] += 1

            if old_phase != phase:
                old_phase = phase
                if type == 'ADPH':
                    limit = phase + 1


        initial_phase[phase] = 1
    return initial_phase, path_matrix, ln_likelihood


def phase_sample_full(initial_states, transitions, length):
    """
    Generates a random sample from an ADPH
    :param initial_states:      an array of initial state probabilities
    :param transitions:         an array for the sub-stochastic transition matrix
    :param length:              The number of samples required
    :return:                    A list of sojourn times, the sum of individual path matrices and initial states
    """

    # Initialise the output lists and transition probabilities
    number_of_phases = len(initial_states)
    path_matrix = np.zeros((number_of_phases, number_of_phases + 1))
    initial_phase = np.zeros(number_of_phases)
    times = []
    b = 1 - transitions.sum(axis=1)
    probability_matrix = np.column_stack((transitions, b))

    # Iterate over the number of samples required
    for sample in range(length):
        sojourn = 0
        path = np.zeros((number_of_phases, number_of_phases + 1))

        # Choose a random starting phase
        phase = np.random.choice(number_of_phases, 1, False, initial_states)[0]
        old_phase = phase

        # Update the count of initial starting phases
        initial_phase[phase] += 1

        # Get the probabilities associated with the starting phase
        probabilities = probability_matrix[phase]

        # Follow the sample path through the state transitions.
        while phase < number_of_phases:

            # Check if a phase transition has occurred
            if phase != old_phase:

                # If a phase transition has occurred update the probabilities with those of the new phase
                probabilities = probability_matrix[phase]
                old_phase = phase

            sojourn += 1

            # Choose the next phase
            phase = np.random.choice(number_of_phases + 1, 1, False, probabilities)[0]

            # Update the count of phase transitions
            path[old_phase, phase] += 1

        # Update the path matrix with the observed transition counts
        path_matrix += path
        times += [sojourn]

        # Print progress
        stdout.write("\rSample : %5d" % (sample + 1))
        stdout.flush()

    stdout.write("\nFinished!\n")

    return times, initial_phase, path_matrix


def iscf1(initial_states, transitions):
    """
    Tests if an ADPH representation is in canonical form 1
    :param initial_states:  A probability vector of initial states
    :param transitions:     A sub-stochastic transition matrix
    :return:                Boolean indicating whether the definition is in CF1 form
    """
    grok = True
    diags = np.diag(transitions)

    # Get absorption probabilities for all states but the last
    prabs = 1 - transitions.sum(axis=1)[:-1]

    # Check the diagonals are in a descending order
    if (diags!=np.sort(diags)[:: -1]).any():
        grok = False
    # Todo: should also check for appropriate zero values in the upper and lower triangles

    # Check if the absorption probabilities are 0
    if (prabs!=0).any():
        grok = False
    return grok


def iscf3(initial_states, transitions):
    """
    Tests if an ADPH representation is in canonical form 1
    :param initial_states:  A probability vector of initial states
    :param transitions:     A sub-stochastic transition matrix
    :return:                Boolean indicating whether the definition is in CF3 form
    """
    grok = True
    diags = np.diag(transitions)

    #Check is the diagonals are in ascending order
    if (diags != np.sort(diags)).any():
        grok = False
    # Todo: should also check for appropriate zero values in the upper and lower triangles

    # Check if the process starts only in the first state
    if (initial_states[0] != 1) or (initial_states[1:] != 0):
        grok = False
    return grok


def cf1to3(initial_states, transitions):
    """
    Converts a transition matrix in canonical form 3 into canonical form 1
    :param transitions: The sub-transition matrix in canonical form 3
    :param initial_states: The initialisation vector for canonical form 3
    :return:
    """
    phases = len(initial_states)
    new_transitions = np.diag(np.diag(transitions)[::-1])
    new_initials = np.zeros_like(initial_states, dtype=float)
    s = np.cumsum(initial_states)
    new_initials[0] = 1
    for i in range(phases - 1):
        new_transitions[i, i + 1] = (1 - new_transitions[i, i]) * s[phases-2-i] / s[phases-1-i]
    return new_initials, new_transitions


def cf3to1(initial_states, transitions):
    """
    Converts a transition matrix in canonical form 1 into canonical form 3
    :param transitions: The sub-transition matrix in canonical form 1
    :param initial_states: The initialisation vector for canonical form 1
    :return:
    """
    phases = np.shape(transitions)[0]
    new_transitions = np.diag(np.diag(transitions)[::-1])
    new_initials = np.zeros_like(initial_states, dtype=float)
    b = 1 - transitions.sum(axis=1)
    baz = 1
    for i in range(phases - 1):
        new_transitions[i, i + 1] = 1 - new_transitions[i, i]
        new_initials[phases - i - 1] = baz * b[i] / (1 - transitions[i, i])
        baz += -new_initials[phases - i - 1]
    new_initials[0] = 1 - new_initials.sum()
    return new_initials, new_transitions


def tocf1(initial_states, transitions):
    """
    Converts and arbitrary ADPH into canonical form 1
    :param transitions: An upper triangle sub-transition matrix
    :param initial_states: The initial state vector
    :return: A sub-transition matrix and initial state vector in canonical form 1
    """

    def get_elementary_series():
        """
        All paths through the ADPH
        :return: A list of tuples containing the state indices of each path
        """
        phase_list = list(range(phases))
        state_indices = []
        for i in range(1, len(phase_list) + 1):
            state_indices += [tuple(list(a) + [phases]) for a in itertools.combinations(phase_list, i)]
        return state_indices

    def series_probability():
        """
        Determine the probability for each path in the ADPH.
        :return: A dictionary of path probabilities indexed by the path tuple.
        """
        series_prob = {}
        for path in paths:
            probability = 1

            # Compute the product of transition probabilities given transition to the next state in the tuple i
            for j in range(len(path) - 1):
                probability *= probability_matrix[path[j], path[j + 1]] / (1 - probability_matrix[path[j], path[j]])

            # Multiply the path probability by the probability of starting in state i[0]
            series_prob[path] = probability * initial_states[path[0]]
        return series_prob

    def absorbing_vector():
        """
        Define the binary vectors associated with each path.
        :return: A dictionary of binary vectors indexed by the absorbing path tuple.
        """
        diagonals = np.diag(transitions)
        diagonal_sort = np.argsort(diagonals)[::-1]
        ranks = np.arange(phases)[diagonal_sort.argsort()]
        binary_vectors = {}

        # Iterate over all absorbing path_matrix
        for path in paths:
            vec = [0] * phases

            # Iterate over the states in an absorbing path.
            for j in range(len(path) - 1):
                vec[ranks[path[j]]] = 1
            binary_vectors[path] = vec
        return binary_vectors

    def basic_mix(binary_vector, weight, diags, basicdict):
        """
        Recursive function to compute the mix of basic vectors which are equivalent to vec
        :param binary_vector: A binary vector as a list
        :param weight: A probability weight.
        :param diags: The sorted eigenvalues of an ADPH
        :param basicdict: The dictionary of basic path probabilities to update
        :return: An updated version of basicdict.
        """

        def find10():
            """
            Find the location in a binary vector where it will be split.  This is only called in the vector is not basic.
            :return: The location in binary_vector where 1 is followed by 0
            """
            for position in range(len(binary_vector) - 1):
                if binary_vector[position] == 1 and binary_vector[position + 1] == 0:
                    return position

        # Test if the vector supplied is a basic vector
        value = 0
        is_basic = False
        for i, j in enumerate(binary_vector[::-1]):
            value += j * 2 ** i
        if 2 ** sum(binary_vector) - 1 == value:
            is_basic = True

        # If it is a basic vector add its weight to the dictionary
        if is_basic:
            basicdict[value] += weight

        # If it's not a basic vector then compute the mix of absorbing paths which are equivalent.
        elif weight > 0:
            location = find10()
            qi = (1 - diags[location]) / (1 - diags[location + 1])
            pi = 1 - qi
            vector_1 = binary_vector.copy()
            vector_1[location + 1] = 1
            vector_2 = vector_1.copy()
            vector_2[location] = 0
            basic_mix(vector_2, weight * qi, diags, basicdict)
            basic_mix(vector_1, weight * pi, diags, basicdict)
        return basicdict

    # Make sure the matrix supplied is upper triangle.
    if np.allclose(transitions, np.triu(transitions)):

        # Compute the revised transition matrix
        phases = len(initial_states)
        b = 1 - transitions.sum(axis=1)
        probability_matrix = np.column_stack((transitions, b))
        new_transitions = np.diag(np.sort(np.diag(transitions))[::-1])
        for i in range(phases - 1):
            new_transitions[i, i + 1] = 1 - new_transitions[i, i]

        # Compute the revised initial state vector
        diags = np.diag(new_transitions).tolist()
        paths = get_elementary_series()
        weights = series_probability()
        vectors = absorbing_vector()
        mix_dictionary = {2 ** (i + 1) - 1: 0 for i in range(phases)}
        for i in paths:
            mix_dictionary = basic_mix(vectors[i], weights[i], diags, mix_dictionary)
        mix_keys = list(mix_dictionary.keys())
        mix_keys.sort(reverse=True)
        new_initials = np.array([mix_dictionary[a] for a in mix_keys])
        return new_initials, new_transitions

    # The case where the matrix supplied is not upper triangle.
    else:
        print('Matrix needs to be upper triangle.')


def import_data(file):
    """
    Imports a file of transfusion data and returns it as a list where each list entry represents the number of units
    transfused into a given patient.
    :param file: A csv file in the form units, number of patients.
    :return: A list of transfusion quantities.
    """
    with open(file, 'r') as f:
        output = [item for sublist in [[a[0]] * a[1] for a in [[int(b) for b in line.strip('\n').split(sep=',')]
                                                               for line in f.readlines()]] for item in sublist]
    f.close()
    return output


def phase_sample(initial_states, transitions, length):
    """
    Generates a sample from a discrete phase-type distribution.  This runs about three times faster than dph_sample as
    it does not record the path through the process.
    :param initial_states:      The initial state probabilities
    :param transitions:         A sub-stochastic transition matrix
    :param length:              The number of samples required
    :return: A matrix of sojourn times.
    """
    eye = np.eye(transitions.shape[0])
    one = np.ones((transitions.shape[0], 1))
    times = []
    for sample in range(length):
        foo = np.random.rand()
        time = 0
        prob = 0
        while prob <= foo:
            time += 1
            prob = initial_states @ (eye - np.linalg.matrix_power(transitions, time)) @ one
        times += [time]

        # Print progress
        stdout.write("\rSample : %5d" % (sample + 1))
        stdout.flush()

    stdout.write("\nFinished!\n")
    return times


def flog(val):
    if val == 1. or val == 0.:
        out = 0
    else:
        out = val * np.log(val) + (1 - val) * np.log(1 - val)
    return out


def tukey(grades):
    x = []
    y = []

    # proportion beyond and including the grade
    alpha = np.array([grades[0: a].sum() for a in range(len(grades))])

    # proportion beyond the grade
    beta = np.array([alpha[a + 1] for a in range(len(grades) - 1)] + [1.])

    # phi(alpha)
    phi1 = np.array([flog(a) for a in alpha])

    # phi(beta)
    phi2 = np.array([flog(a) for a in beta])

    for i in range(len(grades)):
        if alpha[i] - beta[i] != 0:
            x += [(phi1[i] - phi2[i]) / (alpha[i] - beta[i])]
            y += [grades[i]]

    x = np.array(x)
    y = np.array(y)
    return x, y