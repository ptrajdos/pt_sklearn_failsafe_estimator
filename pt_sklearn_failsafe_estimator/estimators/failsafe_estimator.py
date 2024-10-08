from sklearn.base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import logging

class FailsafeEstimator(BaseEstimator, ClassifierMixin):
    """
    Class that prevents the model from failing during training phase.

    If base model raise an Exception, then it is replaced with the default estimator.

    """
    def __init__(self, base_estimator = None, default_estimator = None, exceptions = None, under_test=True):
        self.base_estimator = base_estimator
        self.default_estimator = default_estimator
        self.exceptions = exceptions
        self.under_test = under_test
    
    @staticmethod
    def _get_default_model():
        return KNeighborsClassifier()

    def fit(self, X, y):
        base_estimator_ = clone(self.base_estimator) if self.base_estimator is not None\
            else FailsafeEstimator._get_default_model()
        
        exc_to_catch = tuple([e for e in self.exceptions]) if self.exceptions is not None\
            else (ValueError,)
        if not self.under_test:
            try:
                #TODO what if it should fail?
                base_estimator_.fit(X, y)
            except exc_to_catch as e:
                logging.warning("Failsafe Estimator -- falling back to default estimator!")
                base_estimator_ = clone(self.default_estimator) if self.default_estimator is not None\
                    else DummyClassifier()
                
                base_estimator_.fit(X,y)
        else:
            base_estimator_.fit(X, y)

        self.base_estimator_ = base_estimator_
        return self
    
    def predict(self, X):
        check_is_fitted(self, ("base_estimator_"))
        return self.base_estimator_.predict(X)
    
    def __getattr__(self, name):
        """
        Redirects any method calls or attribute accesses to the wrapped estimator.
        """
        has_attr = 'base_estimator_' in self.__dict__
        
        if has_attr:
            return getattr(self.base_estimator_, name)
        elif self.base_estimator is not None: 
            return getattr(self.base_estimator, name)
        else: 
            return getattr(FailsafeEstimator._get_default_model(), name)
