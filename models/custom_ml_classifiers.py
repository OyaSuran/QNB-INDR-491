
import numpy  as np

from enum import Enum
from models.custom_ml import CustomML

from sklearn.utils.fixes     import loguniform

from sklearn.metrics         import f1_score
from sklearn.metrics         import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold

from sklearn.svm             import SVC
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier

from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier


class CustomMLSVC(CustomML):
    
    
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
                
        # create object from model
        self.model = model(C=1.0, kernel='linear', degree=3, 
                           gamma='scale', coef0=0.0, shrinking=True,
                           probability=True, tol=0.001, cache_size=200,
                           verbose=False, max_iter=-1,
                           decision_function_shape='ovr', break_ties=False, random_state=None)
        
        
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        
        # create ranges for different parameter of model
        distributions = {'C': loguniform(1e0, 1e3), 'gamma': loguniform(1e-4, 1e-3), 
                         'kernel': ['rbf', 'poly', 'sigmoid'],}
        
        # search best parameters via random search
        self.model   = RandomizedSearchCV(self.model, distributions, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)
        
        
            
    
class CustomMLGaussianNB(CustomML):

    
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
        
        # create object from model
        self.model = model(priors=None, var_smoothing=1e-09)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        # create ranges for different parameter of model
        distributions = {'var_smoothing': loguniform(1e-13, 1e-07)}
        
        # search best parameters via random search
        self.model   = RandomizedSearchCV(self.model, distributions, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)
    

class CustomMLDecisionTree(CustomML):
        
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
        
        # create object from model
        self.model = model(criterion='gini', splitter='best', max_depth=None,
                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                           max_features=None, random_state=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, ccp_alpha=0.0)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        
        # create ranges for different parameter of model
        distributions = {
                         
                         'criterion'        : ['gini', 'entropy'],
                         'splitter'         : ['best', 'random'],
                         'min_samples_split': list(range(1, 10)),
                         'min_samples_leaf' : list(range(2, 100, 2)),
                         'max_depth'        : list(range(2, 100, 2)),
                         'max_features'     : ['auto', 'sqrt'],
                        }
        
        # search best parameters via random search
        self.model   = RandomizedSearchCV(self.model, distributions, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)
        
        
    
    
class CustomMLKNeighbors(CustomML):
        
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
        
        # create object from model
        self.model = model(n_neighbors=2, weights='uniform',
                           algorithm='auto', leaf_size=30, p=2,
                           metric='minkowski', metric_params=None, n_jobs=None)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        # create ranges for different parameter of model
        distributions = {'leaf_size'   : list(range(10, 50)),
                         'p'           : (1, 2),
                         'weights'     : ('uniform', 'distance'),
                         'metric'      : ('minkowski', 'chebyshev')}
        
        # search best parameters via random search
        self.model   = RandomizedSearchCV(self.model, distributions, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)
        

    

class CustomMLLogisticReg(CustomML):
        
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
                
        # create object from model
        self.model = model(penalty='l2', dual=False, tol=0.0001,
                           C=1.0, fit_intercept=True, intercept_scaling=1,
                           random_state=None, solver='lbfgs',
                           max_iter=100, multi_class='auto', verbose=0, 
                           warm_start=False)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        
        # define search space
        parameters = {}
        parameters['solver']        = ['lbfgs']
        parameters['fit_intercept'] = [True, False]
        parameters['penalty']       = ['l2']
        parameters['C']             = [0.3, 0.1, 1, 3]
        parameters['max_iter']      = list(range(100, 1000, 100))
        
        # define search
        self.model = RandomizedSearchCV(self.model, parameters, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)
    

class CustomMLRandomForest(CustomML):
        
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
                
        # create object from model
        self.model = model(n_estimators=100, criterion='gini', max_depth=None,
                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                           max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                           bootstrap=True, oob_score=False,
                           verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        # create ranges for different parameter of model
        distributions = {'n_estimators'     : list(range(100, 1000, 50)),
                         'min_samples_leaf' : list(range(1, 100)),
                         'max_depth'        : list(range(2, 100)),
                         'max_features'     : ['auto', 'sqrt'],
                         'bootstrap'        : [True, False]}
        
        # search best parameters via random search
        self.model   = RandomizedSearchCV(self.model, distributions, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)
    

class CustomMLXGB(CustomML):
        
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
                        
        # create object from model
        self.model = model(objective='binary:logistic', use_label_encoder=False)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):
        #https://xgboost.readthedocs.io/en/stable/parameter.html
        #https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        
        params = {}
        params['objective']         = 'binary:logistic'
        params['use_label_encoder'] = False
        
        model_to_tune = self.model_class(*params)
        
        parameters = {
            'max_depth'        : list(range(2, 10, 2)),
            'min_child_weight' : list(range(1, 6,  2)),
            'gamma'            : [i/10.0 for i in range(0,5)],
            'objective'        : ['binary:logistic'],
            'use_label_encoder': [False]
        }
        
        self.model = RandomizedSearchCV(self.model, parameters, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)

    
class CustomMLLightGBM(CustomML):
        
    def __init__(self, model, metrics: dict = {"f1" : f1_score, "roc_auc" : roc_auc_score}):
        """
            
        Initialize model with defaut parameters.
        
        """
        super().__init__(model, metrics)
                
        params={}
        params['learning_rate'] = 0.03
        params['boosting_type'] = 'gbdt'
        params['objective']     = 'binary'
        params['metric']        = 'binary_logloss'
        params['max_depth']     = 10 
        
        # create object from model
        self.model = self.model_class(**params)
    
    
    def hyperparams_search(self, X: np.ndarray, y: np.ndarray, n_iter = 100, cv=5):

        #https://neptune.ai/blog/lightgbm-parameters-guide
        #https://github.com/Microsoft/LightGBM/issues/1339
        #https://towardsdatascience.com/hyper-parameter-tuning-in-python-1923797f124f
        """
        
        Hyperparameter search for the self.model.
        
        :param X  : test input features data
        :param y  : test input target data
        
        :return   : score for each metric
        
        """
        
        params={}
        params['learning_rate']   = 0.03
        params['boosting_type']   = 'gbdt'
        params['objective']       = 'binary'
        params['metric']          = 'auc'
        params['num_boost_round'] = 100
        
        model_to_tune = self.model_class(*params)

        parameters = {
                      'num_leaves'             : [20, 40, 60, 80, 100], 
                      'learning_rate'          : [0.03, 0.01, 0.1, 0.3],
                      'min_child_samples'      : [5, 10, 15],
                      'max_depth'              : [-1, 5, 10, 20],
                      'learning_rate'          : [0.05, 0.1, 0.2], 
                      'reg_alpha'              : [0, 0.01, 0.03],
                      'num_boost_round'        : list(range(30, 100, 10)),
                      'boosting_type'          : ['gbdt'],
                      'objective'              : ['binary'],
                      'metric'                 : ['auc'],
                     }

        self.model = RandomizedSearchCV(model_to_tune, parameters, n_iter=n_iter, cv=cv, verbose=2).fit(X, y)



class CustomMLClassifiers(Enum):
    
    GAUSSIANNB   = CustomMLGaussianNB(model=GaussianNB)
    DecisionTree = CustomMLDecisionTree(model=DecisionTreeClassifier)
    # KNeighbors   = CustomMLKNeighbors(model=KNeighborsClassifier)
    LogisticReg  = CustomMLLogisticReg(model=LogisticRegression)
    RandomForest = CustomMLRandomForest(model=RandomForestClassifier)
    XGBoost      = CustomMLXGB(model=XGBClassifier)
    LightGBM     = CustomMLLightGBM(model=LGBMClassifier)
    # SVC        = CustomMLSVC(model=SVC)
    
    
    @classmethod
    def list(cls, set_class_weight: bool = False):
        
        classifiers = list(map(lambda c: c, cls))
        for classifier in classifiers:
            classifier.value.set_class_weight = set_class_weight
        
        return classifiers


