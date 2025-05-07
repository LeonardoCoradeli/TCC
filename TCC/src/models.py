import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def get_random_forest(random_state=42, cost_sensitive=False):
    params = {'n_estimators': 100, 'random_state': random_state}
    if cost_sensitive:
        params['class_weight'] = 'balanced'
    return RandomForestClassifier(**params)

class TorchClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        cost_sensitive=False,
        device='cpu'
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.cost_sensitive = cost_sensitive
        self.device = device
        self._built = False

    def _build_model(self, y_train):
        self.model_ = SimpleMLP(self.input_dim, self.hidden_dim).to(self.device)
        if self.cost_sensitive:
            neg, pos = np.bincount(y_train)
            pos_weight = torch.tensor(neg/pos, dtype=torch.float32).to(self.device)
            self.criterion_ = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion_ = nn.BCEWithLogitsLoss()
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        self._built = True

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if not self._built:
            self._build_model(y)
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                self.optimizer_.zero_grad()
                logits = self.model_(xb)
                loss = self.criterion_(logits, yb)
                loss.backward()
                self.optimizer_.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.vstack([1-probs, probs]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:,1]
        return (proba > 0.5).astype(int)

def get_catboost(random_state=42, cost_sensitive=False, cat_features=[]):
    params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': random_state,
        'cat_features': cat_features
    }
    if cost_sensitive:
        params['class_weights'] = 'balanced'
    
    model = CatBoostClassifier(**params)
    return model