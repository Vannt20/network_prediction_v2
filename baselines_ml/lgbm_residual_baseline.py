import os
import joblib
import numpy as np
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time
from features.feature_store import to_relative_features


class LGBMResidualBaseline:
    """
    LightGBM-Residual (nhánh học máy thứ hai của ST-Adaptive-Ensemble v3):
    - Cùng kho đặc trưng shared_model với các GBDT khác, nhưng đặc trưng được đưa về dạng
      tương đối so với lag_1 (to_relative_features).
    - Mục tiêu là phần dư Delta = y_t - y_{t-1}; dự báo y_hat = lag_1 + f(x).
    - Vì học mức thay đổi thay vì mức tuyệt đối, mô hình không bị chặn bởi miền giá trị
      Min-Max của tập Train (khắc phục giới hạn ngoại suy của mô hình cây).
    - flow_id (cột 0) là native categorical feature; tăng trưởng cây theo lá (leaf-wise).
    - subsample / colsample_bytree < 1 để các seed cho ra các mô hình khác nhau.
    """
    def __init__(self, feature_names=None, objective='regression', num_leaves=31,
                 learning_rate=0.05, n_estimators=1000, early_stopping_rounds=30,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, **kwargs):
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.params = {
            'objective': objective,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'subsample_freq': 1,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbose': -1,
            **kwargs
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def _transform(self, X):
        if self.feature_names is None:
            raise ValueError("LGBMResidualBaseline cần feature_names để xác định cột lag_1.")
        return to_relative_features(np.asarray(X), self.feature_names)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        R_tr, l1_tr = self._transform(X_train)
        self.model = lgb.LGBMRegressor(**self.params)

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            R_va, l1_va = self._transform(X_val)
            eval_set = [(R_va, np.asarray(y_val) - l1_va)]
            if self.early_stopping_rounds and self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=100))

        self.model.fit(
            R_tr, np.asarray(y_train) - l1_tr,
            eval_set=eval_set,
            categorical_feature=[0],
            callbacks=callbacks
        )
        return self

    def predict(self, X):
        R, l1 = self._transform(X)
        preds = l1 + self.model.predict(R)
        return np.clip(preds, 0.0, None)

    def evaluate(self, X_test, y_test, batch_size=64):
        preds = self.predict(X_test)
        metrics = calc_metrics_numpy(preds, y_test)
        inf_time = measure_inference_time(lambda b: self.predict(b), X_test, batch_size=batch_size)
        metrics['inference_time_ms'] = inf_time
        return metrics, preds

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, filepath)

    def load(self, filepath):
        obj = joblib.load(filepath)
        self.model = obj['model']
        self.feature_names = obj['feature_names']
        return self
