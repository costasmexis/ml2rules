import numpy as np
import pandas as pd
import re
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

SEED = 42

def split_string(s):
    return re.split(' >=| <=| >| <', s)

class MyClass:
    def __init__(self, df: pd.DataFrame, target: str, tree_clf: DecisionTreeClassifier):
        self.tree_clf = tree_clf
        self.target = target
        
        self.feature_names = df.drop(target, axis=1).columns.values.tolist()
        self.class_names = df[target].unique().tolist()
        
        self.rules = None
        self.cleaned_rules = None

    
    def get_rules(self):
        """
        Generate a list of rules from a decision tree.

        Args:
            tree (sklearn.tree.DecisionTreeClassifier): The decision tree model.
            feature_names (list): List of feature names.
            class_names (list): List of class names.

        Returns:
            list: List of rules generated from the decision tree.
        """
        tree_ = self.tree_clf.tree_
        feature_name = [
            self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []
        def recurse(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        self.rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if self.class_names is None:
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {self.class_names[1-l>0]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            self.rules += [rule]

    def clean_rules(self):
        """
        Cleans the given rules by removing unnecessary characters and replacing column names with their corresponding values.

        Arges:
            rules (list): A list of rules to be cleaned.
            X (pd.DataFrame): The input DataFrame containing the column names.

        Returns:
            list: A list of cleaned rules.
        """
        self.cleaned_rules = []
        for rule in self.rules:
            rule = rule.replace('if ', '').replace(' and ', ', ')
            rule = rule.replace('(', '').replace(')', '')
            rule = rule.split(' then')[0]
            rule = rule.split(', ')
            for i, r in enumerate(rule):
                __r = split_string(r)[0]
                for col in self.feature_names:
                    if col == __r:
                        rule[i] = f'"{col}"]{r.split(col)[1]}'
            self.cleaned_rules.append(rule)

    def rule_to_python(self, X_test: pd.DataFrame, idx: int = 0):
        """
        Convert a rule to Python code.

        Args:
            cleaned_rules (list): A list of cleaned rules.
            df (pd.DataFrame): The input DataFrame to sample rows based on rule.
            idx (int, optional): The index of the rule to convert. Defaults to 0.

        Returns:
            list: A list of Python code representing the rule.
        """
        command = []
        for r in self.cleaned_rules[idx]:
            command.append(f'(X_test[{r})')

        command = ' & '.join(command)
        eval(f'X_test[({command})]')
        index = eval(f'X_test[({command})]').index
        return index.values

