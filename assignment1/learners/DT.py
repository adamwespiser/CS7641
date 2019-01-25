import numpy as np
from sklearn import tree
import sklearn.model_selection as ms
from sklearn.tree import DecisionTreeClassifier as SK_DT


import learners

class dtclf_pruned(SK_DT):
    def remove_subtree(self,root):
        '''Clean up'''
        tree = self.tree_
        visited,stack= set(),[root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left =tree.children_left[v]
            right=tree.children_right[v]
            if left >=0:
                stack.append(left)
            if right >=0:
                stack.append(right)
        for node in visited:
            tree.children_left[node] = -1
            tree.children_right[node] = -1
        return 
        
    def prune(self):      
        C = 1-self.alpha
        if self.alpha <= -1: # Early exit
            return self
        tree = self.tree_        
        bestScore = self.score(self.valX,self.valY)        
        candidates = np.flatnonzero(tree.children_left>=0)
        for candidate in reversed(candidates): # Go backwards/leaves up
            if tree.children_left[candidate]==tree.children_right[candidate]: # leaf node. Ignore
                continue
            left = tree.children_left[candidate]
            right = tree.children_right[candidate]
            tree.children_left[candidate]=tree.children_right[candidate]=-1
            score = self.score(self.valX,self.valY)
            if score >= C*bestScore:
                bestScore = score                
                self.remove_subtree(candidate)
            else:
                tree.children_left[candidate]=left
                tree.children_right[candidate]=right
        assert (self.tree_.children_left>=0).sum() == (self.tree_.children_right>=0).sum() 
        return self
        
    def fit(self,X,Y,sample_weight=None,check_input=True, X_idx_sorted=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) 
        self.trgX = X.copy()
        self.trgY = Y.copy()
        self.trgWts = sample_weight.copy()
        sss = ms.StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=123)
        for train_index, test_index in sss.split(self.trgX,self.trgY):
            self.valX = self.trgX[test_index]
            self.valY = self.trgY[test_index]
            self.trgX = self.trgX[train_index]
            self.trgY = self.trgY[train_index]
            self.valWts = sample_weight[test_index]
            self.trgWts = sample_weight[train_index]
        super().fit(self.trgX,self.trgY,self.trgWts,check_input,X_idx_sorted)
        self.prune()
        return self
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0,
                 class_weight=None,
                 presort=False,
                 alpha = 0):
        super(dtclf_pruned, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            presort=presort)
        self.alpha = alpha

    def numNodes(self):
        assert (self.tree_.children_left>=0).sum() == (self.tree_.children_right>=0).sum() 
        return (self.tree_.children_left>=0).sum()

class DTLearner(learners.BaseLearner):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0,
                 class_weight=None,
                 presort=False,
<<<<<<< HEAD
                 alpha = 0,
                 verbose=False):
        super().__init__(verbose)
        self._learner = dtclf_pruned(
=======
                 alpha=0,
                 verbose=False):
        super().__init__(verbose)
        self._alpha = alpha
        self.value_x = None
        self.value_y = None
        self.training_x = None
        self.training_y = None
        self.value_weights = None
        self.training_weights = None

        self._learner = tree.DecisionTreeClassifier(
>>>>>>> 0a6cd1d65d14eaecac766f3659e95b1f02ad60aa
                 criterion=criterion,
                 splitter=splitter,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 class_weight=class_weight,
                 alpha=alpha,
                 presort=presort)

    def numNodes(self):
        self._learner.numNodes()

    def learner(self):
        return self

    @property
    def classes_(self):
        return self._learner.classes_

    @property
    def n_classes_(self):
        return self._learner.n_classes_

    # For pruning we need to pass alpha around
    def get_params(self, deep=True):
        """
        Get the current parameters for the learner. This passes the call back to the learner from learner()

        :param deep: If true, fetch deeply
        :return: The parameters
        """
        extra_params = {'alpha': self._alpha}
        params = self._learner.get_params(deep)

        return {k: v for d in (params, extra_params) for k, v in d.items()}

    def set_params(self, **params):
        """
        Set the current parameters for the learner. This passes the call back to the learner from learner()

        :param params: The params to set
        :return: self
        """
        if 'alpha' in params:
            self._alpha = params.pop('alpha', None)

        return self._learner.set_params(**params)

    # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/helpers.py
    # Based on the pruning technique from Mitchell
    def remove_subtree(self, root):
        """
        Clean up
        :param root:
        :return:
        """
        tmp_tree = self._learner.tree_
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left = tmp_tree.children_left[v]
            right = tmp_tree.children_right[v]
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)
        for node in visited:
            tmp_tree.children_left[node] = -1
            tmp_tree.children_right[node] = -1
        return

    def prune(self):
        c = 1 - self._alpha
        if self._alpha <= -1:  # Early exit
            return self
        tmp_tree = self._learner.tree_
        best_score = self.score(self.value_x, self.value_y)
        candidates = np.flatnonzero(tmp_tree.children_left >= 0)
        for candidate in reversed(candidates):  # Go backwards/leaves up
            if tmp_tree.children_left[candidate] == tmp_tree.children_right[candidate]:  # leaf node. Ignore
                continue
            left = tmp_tree.children_left[candidate]
            right = tmp_tree.children_right[candidate]
            tmp_tree.children_left[candidate] = tmp_tree.children_right[candidate] = -1
            score = self.score(self.value_x, self.value_y)
            if score >= c * best_score:
                best_score = score
                self.remove_subtree(candidate)
            else:
                tmp_tree.children_left[candidate] = left
                tmp_tree.children_right[candidate] = right
        assert (self._learner.tree_.children_left >= 0).sum() == (self._learner.tree_.children_right >= 0).sum()

        return self

    def fit(self, x, y, sample_weight=None, check_input=True, x_idx_sorted=None):
        if sample_weight is None:
            sample_weight = np.ones(x.shape[0])
        self.training_x = x.copy()
        self.training_y = y.copy()
        self.training_weights = sample_weight.copy()
        # TODO: Make this tunable? at least random_state?
        sss = ms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
        for train_index, test_index in sss.split(self.training_x, self.training_y):
            self.value_x = self.training_x[test_index]
            self.value_y = self.training_y[test_index]
            self.training_x = self.training_x[train_index]
            self.training_y = self.training_y[train_index]
            self.value_weights = sample_weight[test_index]
            self.training_weights = sample_weight[train_index]
        self._learner.fit(self.training_x, self.training_y, self.training_weights, check_input, x_idx_sorted)
        self.prune()
        return self

    def predict(self, data):
        return self._learner.predict(data)

    def write_visualization(self, path):
        """
        Write a visualization of the given learner to the given path (including file name but not extension)
        :return: self
        """
        return tree.export_graphviz(self._learner, out_file='{}.dot'.format(path))
