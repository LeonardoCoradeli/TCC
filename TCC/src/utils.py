import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import category_encoders as ce

def plot_metric_across_folds(metrics_list, metric, save=True, figures_dir=None):
    folds  = [m['fold'] for m in metrics_list]
    values = [m[metric] for m in metrics_list]
    fig, ax = plt.subplots()
    ax.plot(folds, values, marker='o')
    ax.set_title(f'{metric.capitalize()} por fold')
    ax.set_xlabel('Fold')
    ax.set_ylabel(metric.capitalize())
    ax.grid(True)
    if save:
        path = os.path.join(figures_dir, f"{metric}.png")
        fig.savefig(path, bbox_inches='tight')
        print(f"Salvo: {path}")
    plt.show()
    plt.close(fig)

def plot_all_metrics(metrics_list, save=True, figures_dir=None):
    metrics_keys = [k for k in metrics_list[0].keys() if k not in ('fold','tn','fp','fn','tp')]
    for metric in metrics_keys:
        plot_metric_across_folds(metrics_list, metric, save=save, figures_dir=figures_dir)

def plot_confusion_matrix(metrics, fold=None, save=True, figures_dir=None):
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest')
    title = 'Matriz de Confusão'
    if fold is not None:
        title += f' - Fold {fold}'
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Neg', 'Pos'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Neg', 'Pos'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_ylabel('Rótulo verdadeiro')
    ax.set_xlabel('Rótulo previsto')

    if save:
        fname = 'confusion_matrix'
        if fold is not None:
            fname += f'_fold_{fold}'
        path = os.path.join(figures_dir, f"{fname}.png")
        fig.savefig(path, bbox_inches='tight')
        print(f"Salvo: {path}")
    plt.show()
    plt.close(fig)

def save_metrics_to_json(metrics_list, filename='metrics.json', json_dir=None):
    for k, v in metrics_list.items():
        if isinstance(v, np.int64):
           metrics_list[k] = int(v)
        elif isinstance(v, np.float64):
            metrics_list[k] = float(v)

    with open(json_dir+'/'+filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)

    print(f"Métricas salvas em JSON: {json_dir}")

def plot_hyperparam_tuning(
    cv_results: dict,
    param_name: str,
    metric: str = 'mean_test_score',
    save: bool = False,
    tuning_dir: str = None,
    filename: str = None
):
    df = pd.DataFrame(cv_results)
    
    param_col = f'param_{param_name}'
    if param_col not in df.columns:
        raise KeyError(f"Coluna '{param_col}' não encontrada nos resultados de tuning.")
    
    x = df[param_col]
    y = df[metric]

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title(f'{metric} vs {param_name}')
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.grid(True)

    if save and tuning_dir:
        os.makedirs(tuning_dir, exist_ok=True)
        fname = filename or f'{param_name}_{metric}.png'
        path = os.path.join(tuning_dir, fname)
        fig.savefig(path, bbox_inches='tight')
        print(f"Tuning plot salvo em: {path}")

    plt.show()
    plt.close(fig)

def encode_categorical(X, method='onehot'):
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    hash_cols = [col for col in cat_cols if X[col].apply(lambda x: isinstance(x, str) and len(x) == 5).any()]

    if method == 'onehot':
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = ohe.fit_transform(X[cat_cols])

        if hash_cols:
            hash_enc = ce.HashingEncoder(cols=hash_cols)
            X_encoded_hash = hash_enc.fit_transform(X[hash_cols])
            X_encoded = np.hstack([X_encoded, X_encoded_hash])

    elif method == 'label':
        le = LabelEncoder()
        X_encoded = X.copy()
        for col in cat_cols:
            X_encoded[col] = le.fit_transform(X[col])
    else:
        raise ValueError("Método de codificação desconhecido")

    return X_encoded