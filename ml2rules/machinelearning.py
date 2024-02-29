import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier, plot_importance
from tqdm import tqdm
import optuna
from sklearn.model_selection import cross_val_score

SEED = 42

def train_tree_classifier(X_train, y_train):
    """
    Trains a decision tree classifier using the provided training data.
    We want to use a relatively simple model to extract simple rules.
    
    Parameters:
        X_train (array-like): The input features for training.
        y_train (array-like): The target labels for training.
        
    Returns:
        best_estimator (object): The best estimator found by the randomized search.
    """
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    clf = DecisionTreeClassifier(random_state=SEED)
    randomized_search = RandomizedSearchCV(clf, param_grid, n_iter=10, scoring='accuracy', cv=5, random_state=SEED)
    randomized_search.fit(X_train, y_train)
    return randomized_search.best_estimator_

def data_preprocessing(df: pd.DataFrame, target: str, scaler: object = StandardScaler(), test_size: float = 0.33):
    """
    Preprocesses the data by performing the following steps:
    1. Separates the features (X) and the target variable (y) from the input DataFrame.
    2. Splits the data into training and testing sets using stratified sampling.
    3. Applies scaling to the features using the specified scaler object.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        target (str): The name of the target variable column.
        scaler (object, optional): The scaler object used for feature scaling. Defaults to StandardScaler().
        test_size (float, optional): The proportion of the data to be used for testing. Defaults to 0.33.
    
    Returns:
        X_train (pd.DataFrame): The preprocessed training features.
        X_test (pd.DataFrame): The preprocessed testing features.
        y_train (pd.DataFrame): The training target variable.
        y_test (pd.DataFrame): The testing target variable.
    """
    
    X = df.drop(target, axis=1).copy()
    y = df[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=SEED)    
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test =  pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test

def bootstrap_evaluation(df: pd.DataFrame, clf: object, n_iter=100, test_size=0.3):
    """
    Perform bootstrap evaluation on a given DataFrame using a classifier.

    Parameters:
    - df (pd.DataFrame): The input DataFrame. Target variable should be at the LAST COLUMN.
    - clf (object): The classifier object.
    - n_iter (int): The number of iterations for bootstrap evaluation. Default is 100.
    - test_size (float): The proportion of the dataset to use as the test set. Default is 0.3.

    Returns:
    - scores (list): A list of scores obtained from each iteration of bootstrap evaluation.
    """
    values = df.values
    n_size = int(len(df) * test_size)
    scores = []
    for _ in tqdm(range(n_iter)):
        train = resample(values, n_samples=n_size)        
        test = np.array([x for x in values if x.tolist() not in train.tolist()])
        clf.fit(train[:,:-1], train[:,-1])
        score = sklearn.metrics.matthews_corrcoef(test[:,-1], clf.predict(test[:,:-1]))
        scores.append(score)
    return scores

class XGBModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.__estimator = XGBClassifier(random_state=SEED)
        self.model = None    
        
    def train(self, cv: int = 3, scoring: str = 'matthews_corrcoef', n_iter: int = 100):
        """
        Trains an XGBoost classifier using RandomizedSearchCV and cross validation.

        Parameters:
            X_train (array-like): The input features for training.
            y_train (array-like): The target labels for training.

        Returns:
            best_estimator (object): The best estimator found by RandomizedSearchCV.
        """

        param_grid = {
            'booster': ['gbtree', 'gblinear', 'dart'],
            'max_depth': range(3, 11),
            'learning_rate': np.logspace(-3, -1, num=10),
            'subsample': np.linspace(0.5, 1.0, num=10),
            'colsample_bytree': np.linspace(0.5, 1.0, num=10),
            'gamma': np.logspace(-2, 0, num=10),
            'min_child_weight': range(1, 11),
            'n_estimators': range(5, 100, 5),
            'random_state': [SEED]
        }

        randomized_search = RandomizedSearchCV(self.__estimator, param_grid, n_iter=n_iter, scoring=scoring, \
            cv=cv, random_state=SEED, verbose=10, n_jobs=10)
        randomized_search.fit(self.X_train, self.y_train)
        self.model = randomized_search.best_estimator_
        self.model.fit(self.X_train, self.y_train)

        

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)