import numpy as np

import experiments
import learners


class ANNExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/ANN.py
        # Search for good alphas
        alphas = [10 ** -x for x in np.arange(-3, 9.01, 0.5)]
        alphas = [0] + alphas
        # TODO: Allow for better tuning of hidden layers based on dataset provided
        d = self._details.ds.features.shape[1]
        hiddens = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
        learning_rates = sorted([(2**x)/1000 for x in range(8)]+[0.000001])

        params = {'MLP__activation': ['relu', 'logistic', 'tanh'], 'MLP__alpha': alphas,
                  'MLP__learning_rate_init': learning_rates,
                  'MLP__hidden_layer_sizes': hiddens,
                  'MLP__random_state' : [self._details.seed],
                  'MLP__beta_1' : [0.5,0.9,0.99,0.999],
                  'MLP__beta_2' : [0.5,0.9,0.99,0.999]}

        timing_params = {'MLP__early_stopping': False}
        iteration_details = {
            'x_scale': 'log',
            'params': {'MLP__max_iter':
                            [2 ** x for x in range(12)] + [2100, 2400, 2700, 3000]},
            'pipe_params': timing_params
        }
        complexity_param = {'name': 'MLP__alpha', 'display_name': 'Alpha', 'x_scale': 'log',
                            'values': alphas}

        best_params = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # best_params = {'activation': 'relu', 'alpha': 1.0, 'hidden_layer_sizes': (36, 36),
        alpha = 0
        params_wine = {
                'MLP__activation': 'tanh', 
                'MLP__alpha': 0.1,
                'MLP__learning_rate_init': 0.064,
                'MLP__hidden_layer_sizes': (12,12),
                'MLP__beta_1' : 0.99,
                'MLP__beta_2' : 0.99
        }
        if self._details.ds_name == "wine-qual" and self._details.bparams:
            alpha = 0.1
            for k in params.keys():
                if k in params_wine.keys():
                    params[k] = [params_wine.get(k)]
        params_enhancer = {
                'MLP__activation': 'logistic', 
                'MLP__alpha': 0.001,
                'MLP__learning_rate_init': 0.128,
                'MLP__hidden_layer_sizes': (38,38),
                'MLP__beta_1' : 0.5,
                'MLP__beta_2' : 0.999
        }
        if self._details.ds_name == "enhancer-b" and self._details.bparams:
            alpha = 0.001
            for k in params.keys():
                if k in params_enhancer.keys():
                    params[k] = [params_enhancer.get(k)]
        #if self._details.ds_name == "wine-qual":
        #    best_params = params_wine
        #                'learning_rate_init': 0.016}
        # Dataset 2:
        # best_params = {'activation': 'relu', 'alpha': 1e-05, 'hidden_layer_sizes': (16, 16),
        #                'learning_rate_init': 0.064}

        learner = learners.ANNLearner(max_iter=3000, early_stopping=True, random_state=self._details.seed,
                                      verbose=self._verbose)
        if best_params is not None:
            learner.set_params(**best_params)
        cv_best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'ANN', 'MLP',
            params,
            complexity_param=complexity_param,
            seed=self._details.seed,
            timing_params=timing_params,
            iteration_details=iteration_details,
            best_params=best_params,
            threads=self._details.threads, verbose=self._verbose)

        # TODO: This should turn OFF regularization
        of_params = cv_best_params.copy()
        of_params['MLP__alpha'] = alpha
        if best_params is not None:
            learner.set_params(**best_params)
        learner = learners.ANNLearner(max_iter=3000, early_stopping=True, random_state=self._details.seed,
                                      verbose=self._verbose)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
                                       'ANN_', 'MLP', of_params, seed=self._details.seed, timing_params=timing_params,
                                       iteration_details=iteration_details,
                                       best_params=best_params,
                                       threads=self._details.threads, verbose=self._verbose,
                                       iteration_lc_only=True)
