
import numpy  as np
import matplotlib.pyplot as plt 

from sklearn.metrics         import f1_score
from sklearn.metrics         import roc_auc_score
from sklearn.model_selection import train_test_split

from models.custom_ml_classifiers import CustomMLClassifiers


class Classifiers:
    
    
    def __init__(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    test_size    : float,
                    classifiers  : list = CustomMLClassifiers.list(),
                    balance_data : bool = False,
                    scale_pos_weight:int =19,
                    metrics: dict       = {"f1" : f1_score, "roc_auc" : roc_auc_score}) -> None:
        """
        
        Initialize algorithms class. 
        All classifiers in 'classifiers' list will be train and test via this class.
        
        :param metrics     : metrics dict contains name and function for the metric
        :param classifiers : 
        
        """
        
        
        self.X           = X
        self.y           = y
        self.test_size   = test_size
        self.metrics     = metrics
        
        self._train_test_split()
        
        self.is_train     = False        
        self.classifiers  = classifiers
        self.balance_data = balance_data
        
        
    def _train_test_split(self) -> None:
        """
        
        Split dataset into train and test dataset with test size.
        
        """
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                stratify=self.y)
        
        
    def plot_scores(self, scores: dict, title: str) -> None:
        """

        Plot scores for given classifiers.

        :param scores: classifier - metrics dictionary.

        """

        plt.figure(figsize=(16, 8))

        classifiers = [key.name for key in scores["f1"].keys()]

        score_f1  = [scores["f1"][classifier] for classifier in scores["f1"].keys()]
        score_roc = [scores["roc_auc"][classifier] for classifier in scores["roc_auc"].keys()]

        classifiers_axis = np.arange(len(classifiers))

        plt.bar(classifiers_axis - 0.2, score_f1,  0.4, label = 'f1')
        plt.bar(classifiers_axis + 0.2, score_roc, 0.4, label = 'auc')

        for i in range(len(classifiers_axis)):
            plt.text(i - 0.2, score_f1[i]  + 0.01, f"{score_f1[i]:.2f}",  ha = 'center')
            plt.text(i + 0.2, score_roc[i] + 0.01, f"{score_roc[i]:.2f}", ha = 'center')

        plt.xticks(classifiers_axis, classifiers)
        plt.xlabel("Classifiers")
        plt.ylabel("Score")
        plt.title(f"{title} Score for Classifiers")
        plt.legend()
        plt.show()
        
        
    def compare_test(self):
        """
        
        Test all models in classifier list and compare results for test.
        
        :param cv : count of fold for cross validation 
        
        """
        
        if not self.is_train:
            raise Exception("Before test classifiers class 'compare_train' to train classifiers!")
            
        scores = dict()
        
        for classifier in self.classifiers:
               
            print(f"Testing {classifier.name}...")
            try:
                result = classifier.value.test(self.X_test, self.y_test)

                for metric in self.metrics:

                    if metric in scores:
                        scores[metric][classifier] = result[metric]
                    else:
                        scores[metric] = {}
                        scores[metric][classifier] = result[metric]
            except:
                continue
                                
        # display results
        self.plot_scores(scores, title = "Test")
                
        return scores
    
    
    def compare_train(self, cv: int):
        """
        
        Train all models in classifier list and compare results for train.
        
        :param cv : count of fold for cross validation 
        
        """
        
        scores = dict()
        
        for classifier in self.classifiers:
            
            print(f"Training {classifier.name}...")
            try:
                result = classifier.value.train(self.X_train, self.y_train, cv)

                for metric in self.metrics:

                    if metric in scores:
                        scores[metric][classifier] = result[metric]
                    else:
                        scores[metric] = {}
                        scores[metric][classifier] = result[metric]
            except:
                continue
                    
        self.is_train = True
        
        # display results
        self.plot_scores(scores, title = "Train")
        
        return scores


    def hyperparams_search(self, exclude: list):
        """
        
        Hyperparameter search for the all classifiers except exclude list.
                
        
        Comparison of randomized and normal grid search method can be nice to read:
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
        """
        
        classifers_to_tune = set(self.classifiers) - set(exclude)
        
        scores = dict()
        
        for classifier in classifers_to_tune:
               
            print(f"Hyper parameters search for {classifier.name}...", end="")
            classifier.value.hyperparams_search()
            result = classifier.value.test(self.X_test, self.y_test)
            
            for metric in self.metrics:
                
                if metric in scores:
                    scores[metric][classifier] = result[metric]
                else:
                    scores[metric] = {}
                    scores[metric][classifier] = result[metric]
                                
        # display results
        self.plot_scores(scores, title = "Hyper Parameter Search")
                
        return scores
    
    