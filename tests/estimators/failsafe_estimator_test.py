import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from pt_sklearn_failsafe_estimator.estimators.failsafe_estimator import FailsafeEstimator

class FailingEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        raise ValueError("Fit Error!")

class FailsafeEstimatorTest(unittest.TestCase):

    def get_estimators(self):
        return {
            "Base": FailsafeEstimator(),
            "Exc": FailsafeEstimator(exceptions=(ValueError, AttributeError)),
            "Dummy": FailsafeEstimator(default_estimator=DummyClassifier()),
        }
    
    def test_sklearn(self):

        for clf_name, clf in self.get_estimators().items():
            for estimator, check in check_estimator(clf, generate_only=True):
                check(estimator)

    def test_iris_fit_predict_proba(self):

        X, y = load_iris(return_X_y=True)

        estims = self.get_estimators()
        estims_new = dict()

        for k, v in estims.items():
            vc = clone(v)
            vc.under_test = False
            estims_new[f"{k}_n"] = vc

            vf =  clone(v)
            vf.under_test = False
            vf.base_estimator = FailingEstimator()
            estims_new[f"{k}_f"] = vf


        estims.update(estims_new)

        for clf_name, clf in estims.items():
            clf.fit(X,y)

            proba_predicted = clf.predict_proba(X)
            self.assertIsNotNone(proba_predicted, f"Transformed value is None for {clf_name}")
            self.assertIsInstance(proba_predicted, np.ndarray, f"Wrong type of transformed object for {clf_name}")
            self.assertTrue(len(proba_predicted) == len(y), f"Wrong length of the transformed object for {clf_name}")
            self.assertTrue( proba_predicted.shape[1] == len(np.unique(y)), f"Wrong number of columns in transformed object for {clf_name}")
            self.assertFalse(np.isnan(proba_predicted).any(), f"NaNs in prediction for {clf_name}")
            self.assertFalse(np.isinf(proba_predicted).any(), f"Infs in prediction for {clf_name}")
            self.assertTrue( np.min(proba_predicted) >= 0, f"Min value is negative for {clf_name}!")
            self.assertTrue( np.max(proba_predicted) <= 1, f"Max value is greater than one for  {clf_name}!")
            row_sums = np.sum(proba_predicted,axis=1)
            self.assertTrue(np.allclose(row_sums,1.0, atol=1e-9), "Wrong row sums")

            
    def test_iris_pipeline(self):

        X, y = load_iris(return_X_y=True)
         
        estims = self.get_estimators()

        for clf_name, clf in estims.items():
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
            pipeline.fit(X,y)

            pred = pipeline.predict(X)
            self.assertIsNotNone(pred, f"Transformed value is None for {clf_name}")
            self.assertIsInstance(pred, np.ndarray, f"Wrong type of transformed object for {clf_name}")
            self.assertTrue(len(pred) == len(y), f"Wrong length of the transformed object for {clf_name}")
            self.assertTrue( np.allclose(np.unique(pred), np.unique(y)), f"Different set of predicted classes for {clf_name}"  )

    def test_iris_grid_search(self):

        X, y = load_iris(return_X_y=True)

        estims = {
            "Base": FailsafeEstimator(base_estimator= RandomForestClassifier()) ,
        }
         
        for clf_name, clf in estims.items():
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])

            params = {
                'classifier__base_estimator__n_estimators':[5,10,15],
            }
            kappa_scorer = make_scorer(cohen_kappa_score)
            gs = GridSearchCV(pipeline, params, cv=3, scoring=kappa_scorer)

            gs.fit(X,y)

            pred = gs.predict(X)
            self.assertIsNotNone(pred, f"Transformed value is None for {clf_name}")
            self.assertIsInstance(pred, np.ndarray, f"Wrong type of transformed object for {clf_name}")
            self.assertTrue(len(pred) == len(y), f"Wrong length of the transformed object for {clf_name}")
            self.assertTrue( np.allclose(np.unique(pred), np.unique(y)), f"Different set of predicted classes for {clf_name}"  )