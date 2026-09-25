import os
import joblib
import numpy as np
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time


class LGBMBaseline:
    """
    LightGBM Regressor theo chiến lược multi_output: shared_model:
    - 1 model duy nhất cho toàn bộ N luồng OD.
    - flow_id ở cột 0 là native categorical feature.
    - Hàm mất mát L2 / Regression (MSE chuẩn mực tối ưu hồi quy lưu lượng).
    """
    def __init__(self, objective='regression', num_leaves=31,
                 learning_rate=0.05, n_estimators=1000, early_stopping_rounds=30,
                 random_state=42, n_jobs=-1, **kwargs):
        self.params = {
            'objective': objective,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbose': -1,
            **kwargs
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        self.model = lgb.LGBMRegressor(**self.params)
        
        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if self.early_stopping_rounds and self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))

        # flow_id là cột đầu tiên (index 0)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            categorical_feature=[0],
            callbacks=callbacks
        )
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
