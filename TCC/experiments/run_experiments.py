import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.data_loading import load_fraud_data
from src.models import get_random_forest,TorchClassifier, torch
from src.pipeline import CustomPipeline
from src.utils import plot_all_metrics, plot_metric_across_folds, plot_confusion_matrix, save_metrics_to_json
import sys
import os
from typing import Callable, Optional, Tuple
import numpy as np
from src.data_loading import load_fraud_data
from src.models       import get_random_forest, TorchClassifier
from src.pipeline     import CustomPipeline
from src.utils        import (
    plot_confusion_matrix,
    save_metrics_to_json,
    plot_hyperparam_tuning
)
from src.modelTuning import ModelTuning
from src.utils import encode_categorical

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, PROJECT_ROOT)


def run_experiment(X, y, figures_base, json_base, tuning_dir, test_size=0.2, method=None,cost_sensitivity=False,cat_features=[]):

    results = {}

    X_encoded = encode_categorical(X, 'onehot')
    model_rf_enc, grid_rf_enc = ModelTuning.tune_random_forest(cost_sensitive=cost_sensitivity)
    results['RF_enc'] = run_single_model(model_rf_enc, X_encoded, y, 'RF', method, figures_base, json_base, tuning_dir)

    model_cb_cat, grid_cb_cat = ModelTuning.tune_catboost(cat_features=cat_features, ost_sensitive=cost_sensitivity)
    results['CB_cat'] = run_single_model(model_cb_cat, X, y, 'CatBoost', method, figures_base, json_base, tuning_dir)

    model_cb_enc, grid_cb_enc = ModelTuning.tune_catboost(cat_features=cat_features, cost_sensitive=cost_sensitivity)
    results['CB_enc'] = run_single_model(model_cb_enc, X_encoded, y, 'CatBoost', method, figures_base, json_base, tuning_dir)

    model_nn, grid_nn = ModelTuning.tune_torch_mlp(X_encoded, y)
    results['NN_enc'] = run_single_model(model_nn, X_encoded, y, 'NN', method, figures_base, json_base, tuning_dir)

    return results

def run_single_model(model, X, y, model_name, method, figures_base, json_base, tuning_dir):
    metrics = CustomPipeline(model, method=method).train_test_evaluate(X, y)

    save_metrics_to_json(metrics, json_dir=json_base+f'{model_name}/{method}')
    
    plot_confusion_matrix(metrics, figures_dir=figures_base+f'{model_name}/{method}')
    
    if hasattr(model, 'cv_results_'):
        plot_hyperparam_tuning(
            cv_results=model.cv_results_,
            param_name='n_estimators',
            metric='mean_test_score',
            save=True,
            tuning_dir=tuning_dir,
            filename=f'{model.best_estimator_}_tuning_plot.png'
        )

    return metrics
