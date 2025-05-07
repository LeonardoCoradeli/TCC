from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from src.models import get_random_forest, TorchClassifier
from src.utils  import plot_hyperparam_tuning
from sklearn.metrics      import make_scorer, roc_auc_score,f1_score
import numpy as np
from catboost import CatBoostClassifier

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth':    [None, 10, 20],
    }
    base = get_random_forest(random_state=42)
    grid = GridSearchCV(base, param_grid, cv=3, scoring='roc_auc', return_train_score=True)
    grid.fit(X_train, y_train)
    plot_hyperparam_tuning(grid.cv_results_, 'n_estimators')
    return grid.best_estimator_, grid


def tune_torch_mlp(X_train, y_train):
    counts = np.bincount(y_train)
    n_minority = counts[1] if len(counts) > 1 else 0


    n_splits = min(5, max(2, n_minority // 3))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    param_dist = {
        'hidden_dim': [32, 64, 128],
        'lr':         [1e-3, 1e-4],
        'batch_size': [16, 32, 64]
    }

    base = TorchClassifier(
        input_dim      = X_train.shape[1],
        epochs         = 10,
        cost_sensitive = False,
        device         = 'cpu'
    )

    f1_scorer = make_scorer(f1_score, average='binary') 

    search = RandomizedSearchCV(
        estimator           = base,
        param_distributions = param_dist,
        n_iter              = 6,
        cv                  = cv,
        scoring             = f1_scorer,
        return_train_score  = True,
        random_state        = 42,
        n_jobs              = -1,
        error_score         = np.nan
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search

def tune_catboost(X_train, y_train):
    param_grid = {
        'iterations': [500, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8],
    }
    base = CatBoostClassifier(random_seed=42, cat_features=[])
    grid = GridSearchCV(base, param_grid, cv=3, scoring='roc_auc', return_train_score=True)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid