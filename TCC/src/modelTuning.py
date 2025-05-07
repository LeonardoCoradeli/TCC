from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from catboost import CatBoostClassifier
from src.models import get_random_forest, TorchClassifier, get_catboost
import numpy as np

class ModelTuning:
    @staticmethod
    def tune_random_forest(cost_sensitive=False):
        param_grid = {
            'n_estimators': [50, 100, 200, 400],
            'max_depth': [None, 10, 20, 40],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'bootstrap': [True, False]
        }
        base = get_random_forest(random_state=42, cost_sensitive=cost_sensitive)
        grid = GridSearchCV(base, param_grid, cv=3, scoring='roc_auc', return_train_score=True)
        return grid, param_grid


    @staticmethod
    def tune_torch_mlp(X_train, y_train,cost_sensitive=False):
        counts = np.bincount(y_train)
        n_minority = counts[1] if len(counts) > 1 else 0
        n_splits = min(5, max(2, n_minority // 3))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        param_dist = {
            'hidden_dim': [16, 32, 64, 128],
            'lr': [1e-3, 1e-4, 1e-5, 1e-6],
            'batch_size': [16, 32, 64],
            'epochs': [10, 20, 50, 100]
        }
        base = TorchClassifier(input_dim=X_train.shape[1], device='cpu', cost_sensitive=cost_sensitive)
        f1_scorer = make_scorer(f1_score, average='binary')

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=6,
            cv=cv,
            scoring=f1_scorer,
            return_train_score=True,
            random_state=42,
            n_jobs=-1,
            error_score=np.nan
        )
        return search, param_dist

    @staticmethod
    def tune_catboost(cost_sensitive=False,cat_features=[]):
        param_grid = {
            'iterations': [250, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [16, 64]
        }
        base = get_catboost(random_state=42, cat_features=[], cost_sensitive=cost_sensitive)
        grid = GridSearchCV(base, param_grid, cv=3, scoring='roc_auc', return_train_score=True)
        return grid, param_grid