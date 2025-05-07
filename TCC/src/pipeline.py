import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.augmentation import apply_resampling
from src.evaluate import compute_metrics
from sklearn.model_selection import train_test_split

class CustomPipeline:
    def __init__(
        self,
        model,
        method=None,
        folds=5,
        random_state=42,
        cluster_k=5
    ):
        if method is None:
            self.methods = []
        elif isinstance(method, str):
            self.methods = [method]
        else:
            self.methods = list(method)
        
        self.model = model
        self.folds = folds
        self.random_state = random_state
        self.cluster_k = cluster_k

    def fit(self, X, y):
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)

        return self

    def predict(self, X):
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[:, 1]
        else:
            scores = self.model.decision_function(X)
            proba = (scores - scores.min()) / (scores.max() - scores.min())
        preds = (proba > 0.5).astype(int)
        return preds, proba

    def evaluate(self, X, y):
        preds, proba = self.predict(X)
        return compute_metrics(y, preds, proba)
    
    def train_test_evaluate(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            stratify=y, random_state=self.random_state
        )
        
        self.fit(X_train, y_train)
        
        m = self.evaluate(X_test, y_test)
        m.update({'test_size': test_size})
        return m
