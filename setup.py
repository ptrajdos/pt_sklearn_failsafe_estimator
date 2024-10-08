from setuptools import setup, find_packages

setup(
        name='pt_sklearn_failsafe_estimator',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/pt_sklearn_failsafe_estimator',
        description="Estimator that uses default model if fitting of the original model fails",
        packages=find_packages(include=[
                'pt_sklearn_failsafe_estimator',
                'pt_sklearn_failsafe_estimator.*',
                ]),
        install_requires=[ 
                'scikit-learn>=1.2.2',
        ],
        test_suite='test'
        )
