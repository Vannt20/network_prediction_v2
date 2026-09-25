import os
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time


class ExtraTreesBaseline:
    """
    Extra Trees Regressor theo chiến lược multi_output: shared_model:
    - 1 model duy nhất cho toàn bộ N luồng OD.
    - Biến thể ngẫu nhiên hóa mạnh hơn RF, dùng đối chứng khi RF có dấu hiệu quá khớp.
    """
    def __init__(self, n_estimators=200, max_depth=12, min_samples_leaf=5,
                 criterion='squared_error', random_state=42, n_jobs=-1, **kwargs):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'criterion': criterion,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }
        self.model = ExtraTreesRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def evaluate(self, X_test, y_test, batch_size=64):
        preds = self.predict(X_test)
        metrics = calc_metrics_numpy(preds, y_test)
        inf_time = measure_inference_time(lambda b: self.predict(b), X_test, batch_size=batch_size)
        metrics['inference_time_ms'] = inf_time
        return metrics, preds

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)
        return self
