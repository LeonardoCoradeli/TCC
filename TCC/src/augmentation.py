from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.cluster import KMeans


def apply_resampling(X, y, method, random_state=42, cluster_k=5):
    if method == 'ROS':
        sampler = RandomOverSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)

    elif method == 'RUS':
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)

    elif method == 'SMOTE':
        sampler = SMOTE(random_state=random_state, k_neighbors=4)
        X_res, y_res = sampler.fit_resample(X, y)

    elif method == 'ADASYN':
        sampler = ADASYN(random_state=random_state, n_neighbors=4)
        X_res, y_res = sampler.fit_resample(X, y)

    elif method == 'CLUSTER':
        X_min = X[y == 1]
        kmeans = KMeans(n_clusters=cluster_k, random_state=random_state)
        clusters = kmeans.fit_predict(X_min)
        synthetic = []
        for cid in np.unique(clusters):
            X_c = X_min[clusters == cid]
            if X_c.shape[0] > 1:
                sm = SMOTE(random_state=random_state)
                X_syn, _ = sm.fit_resample(X_c, np.ones(len(X_c)))
                synthetic.append(X_syn[len(X_c):])
        X_res = np.vstack([X] + synthetic)
        y_res = np.hstack([y] + [np.ones(s.shape[0]) for s in synthetic])

    elif method == 'COST_SENSITIVE':
        X_res, y_res = X, y

    else:
        raise ValueError(f"Método desconhecido: {method}")

    return X_res, y_res
