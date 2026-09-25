import os
import joblib
import numpy as np
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time


class XGBoostBaseline:
    """
    XGBoost Regressor theo chiến lược multi_output: shared_model:
    - 1 model duy nhất cho toàn bộ N luồng OD.
    - tree_method: hist (tăng tốc độ gấp 10-20 lần và tiết kiệm RAM).
    - Objective: reg:squarederror (Squared Error / MSE chuẩn mực tối ưu hồi quy lưu lượng).
    - flow_id ở cột 0 được phân chia theo ngưỡng số học (numerical split).
    """
    def __init__(self, objective='reg:squarederror',
                 max_depth=6, learning_rate=0.05, n_estimators=1000,
                 subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30,
                 random_state=42, n_jobs=-1, **kwargs):
        self.params = {
            'objective': objective,
            'tree_method': 'hist',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': n_jobs,
            **kwargs
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import xgboost as xgb
        
        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds if (X_val is not None and self.early_stopping_rounds) else None
        )
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
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
        self.model.save_model(filepath)

    def load(self, filepath):
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)
        return self
