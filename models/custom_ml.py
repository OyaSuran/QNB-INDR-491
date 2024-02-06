
import numpy  as np

from time import time
from abc  import ABC, abstractmethod

from sklearn.metrics         import f1_score
from sklearn.metrics         import roc_auc_score
from sklearn.metrics         import confusion_matrix
from sklearn.model_selection import cross_validate


class CustomML(ABC):
    
    
    def __init__(self, model: object, metrics: dict) -> None:
        """
        
        Set any a machine model from scikit-learn, xgboost or lightgbm.
        
        :param model    : any a machine model from scikit-learn, xgboost or lightgbm.
        :param metrics  : metrics dict contains name and function for the metric
        
        """
        
        # set model
        self.model_class = model
        
        self.set_class_weight = False
        
        # set metrics for score
        self.metrics     = metrics
        
        def gini(y, ypreds):
            return 2 * roc_auc_score(y, ypreds) - 1
        
        self.metrics.update({"gini" : gini})
        
        super().__init__()


    def _timer(func):
        
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            
            print(f' executed in {(t2-t1):.1f} seconds.')
            return result
        
        return wrap_func

    
    @_timer
    def train(self, X: np.ndarray, y: np.ndarray, cv: int) -> dict:
        """
        
        Train and test model with cross validate method.
        
        :param X  : train input features data
        :param y  : train input target data
        :param cv : count of fold for cross validation 
        
        :return   : cross validation scores for each metric with mean value
        
        """
        
        # xgboost class weight is different than scikit learn api
        try:
            
            self.class_weight = 19 if self.set_class_weight else None
            self.model.scale_pos_weight = self.class_weight
            
        except:
            pass
        
        try:

            self.class_weight = {0: 1, 1:19} if self.set_class_weight else None
            self.model.class_weight = class_weight
            
        except:
            pass
        
        # return score for each metric
        scores = dict()

        self.model.fit(X, y)
        preds = self.model.predict(X)
        
        # for any case if model returns prob
        # preds = (preds > 0.5).astype(int)
        
        # metrics is dict contains name of metric as key and metric function as value
        for metric in self.metrics:
            
            if metric == "roc_auc" or metric == "gini":
                scores[metric] = self.metrics[metric](y, self.model.predict_proba(X)[:, 1])
            else:
                scores[metric] = self.metrics[metric](y, preds)
        
        
         # cross validate train
#         cv_results = cross_validate(self.model, X, y, 
#                                     cv = cv, 
#                                     scoring = set(self.metrics.keys()), 
#                                     return_train_score=True, return_estimator=True)

#         for metric in self.metrics:
#             scores[metric] = cv_results[f"train_{metric}"].mean()
            
#         self.model = cv_results["estimator"][0]
        
        return scores

    
    @_timer
    def test(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        
        Test model with metrics.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        
        preds = self.model.predict(X)
        
        # for any case if model returns prob
        # preds = (preds > 0.5).astype(int)
       
        # return score for each metric
        scores = dict()
        
        # metrics is dict contains name of metric as key and metric function as value
        for metric in self.metrics:
            
            if metric == "roc_auc" or metric == "gini":
                scores[metric] = self.metrics[metric](y, self.model.predict_proba(X)[:, 1])
            else:
                scores[metric] = self.metrics[metric](y, preds)
        
        print(" ###### Confusion Matrix #######")
        print(confusion_matrix(y, preds))
        print(" ###############################")
            
        return scores
    
    
    @abstractmethod
    def hyperparams_search():
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        
        Comparison of randomized and normal grid search method can be nice to read:
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
        
        """
        pass
    
    