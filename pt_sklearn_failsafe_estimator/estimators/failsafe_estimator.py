from sklearn.base import BaseEstimator, ClassifierMixin, clone, check_is_fitted
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
import logging


def default_exception_check(exception):
    """
    Arguments:

    exception -- exception to be investigated in order to check whether a default model should be used.

    Returns:

    True if default model should be used. False if exception should be raised again

    """
    return True


def default_set_validation(X, y):
    """
    Checks wheter input data is valid.
    If not should raise an exception.

    Arguments:

    X,y  -- dataset in sklearn-compatible format

    """
    pass


class FailsafeEstimator(ClassifierMixin, BaseEstimator):
    """
    Class that prevents the model from failing during training phase.

    If base model raise an Exception, then it is replaced with the default estimator.

    """

    def __init__(
        self,
        base_estimator=None,
        default_estimator=None,
        exceptions=None,
        under_test=True,
        use_default_model_function=None,
        set_validation_function=None,
    ):
        """

        Arguments:

        base_estimator -- estimator to be used

        default_estimator -- estimator to be used when base_estimator fails to fit (throws an exception from list). When other exception is raised it is not catched.

        exceptions -- lis of exceptions to be catched

        under_test:bool -- determines if fitting base_estimator should fail anyway. Used mainly for testing.

        use_default_model_function -- a fuction that checks the exception in a more detailed way to determine whether the default model should be used.

        set_validation_function -- function that checks if the input data is correct.
        """
        self.base_estimator = base_estimator
        self.default_estimator = default_estimator
        self.exceptions = exceptions
        self.under_test = under_test
        self.use_default_model_function = use_default_model_function
        self.set_validation_function = set_validation_function

    @staticmethod
    def _get_default_model():
        return KNeighborsClassifier()

    def _use_default_model(self, exception):
        self.use_default_model_function = (
            default_exception_check
            if self.use_default_model_function is None
            else self.use_default_model_function
        )
        return self.use_default_model_function(exception)

    def _validate_set(self, X, y):
        self.set_validation_function = (
            default_set_validation
            if self.set_validation_function is None
            else self.set_validation_function
        )
        default_set_validation(X,y)

    def fit(self, X, y):
        base_estimator_ = (
            clone(self.base_estimator)
            if self.base_estimator is not None
            else FailsafeEstimator._get_default_model()
        )

        exc_to_catch = (
            tuple([e for e in self.exceptions])
            if self.exceptions is not None
            else (ValueError,)
        )

        if not self.under_test:
            try:
                # TODO what if it should fail?
                self._validate_set(X,y)
                base_estimator_.fit(X, y)
            except exc_to_catch as e:
                if self._use_default_model(e):
                    logging.warning(
                        "Failsafe Estimator -- falling back to default estimator!",
                        exc_info=True,
                    )
                    base_estimator_ = (
                        clone(self.default_estimator)
                        if self.default_estimator is not None
                        else DummyClassifier()
                    )
                else:
                    logging.error(
                        "Failsafe estimator -- exception do not make the estimator to use default model."
                    )
                    raise e

                base_estimator_.fit(X, y)
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
        has_attr = "base_estimator_" in self.__dict__

        if has_attr:
            return getattr(self.base_estimator_, name)
        elif self.base_estimator is not None:
            return getattr(self.base_estimator, name)
        else:
            return getattr(FailsafeEstimator._get_default_model(), name)
