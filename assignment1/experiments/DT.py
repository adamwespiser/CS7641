import numpy as np

import experiments
import learners


class DTExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # TODO: Clean up the older alpha stuff?
        max_depths = np.arange(1, 21, 1)
        alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
        alphas = [x/1000 for x in range(-20,20,2)]

        params = {'DT__criterion': ['gini', 'entropy'],
                  'DT__max_depth': max_depths,
                  'alpha' : alphas,
                  'DT__class_weight': ['balanced', None]
        }  # , 'DT__max_leaf_nodes': max_leaf_nodes}
        params = {'DT__criterion':['gini','entropy'],
                  'DT__alpha':alphas,
                  'DT__class_weight':['balanced']}

        complexity_param = {'name': 'DT__alpha', 
                            'display_name': 'alpha', 
                            'values': alphas}

        best_params = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1:
        # best_params = {'criterion': 'entropy', 'max_depth': 23, 'class_weight': 'balanced'}
        #
        # Dataset 2:
        # best_params = {'criterion': 'entropy', 'max_depth': 4, 'class_weight': 'balanced'}

        learner = learners.DTLearner(random_state=self._details.seed)
        if best_params is not None:
            learner.set_params(**best_params)

        best_params = experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, 'DT', 'DT', params,
                                       complexity_param=complexity_param, seed=self._details.seed,
                                       threads=self._details.threads,
                                       best_params=best_params,
                                       verbose=self._verbose,
                                       apply_pruning=True)

