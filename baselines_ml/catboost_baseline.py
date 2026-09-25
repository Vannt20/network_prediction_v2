import os
import joblib
import numpy as np
from baselines_ml.metrics import calc_metrics_numpy, measure_inference_time


class CatBoostBaseline:
    """
    CatBoost Regressor theo chiến lược multi_output: shared_model:
    - 1 model duy nhất cho toàn bộ N luồng OD.
    - flow_id ở cột 0 là biến phân loại / định danh luồng.
    - Mặc định loss_function='RMSE' tối ưu chuẩn mực cho hồi quy lưu lượng,
      khắc phục triệt để hiện tượng sai số cao của Huber delta=0.01.
    - use_cat_features=False: mặc định tối ưu hóa phân chia ngưỡng số học (tương tự XGBoost),
      tránh overfit do Ordered Target Statistics trên chuỗi thời gian và tăng tốc tối đa.
    """
    def __init__(self, loss_function='RMSE', depth=6,
                 learning_rate=0.05, iterations=1000, early_stopping_rounds=30,
                 use_cat_features=False, random_seed=42, thread_count=-1, **kwargs):
        self.params = {
            'loss_function': loss_function,
            'depth': depth,
            'learning_rate': learning_rate,
            'iterations': iterations,
            'early_stopping_rounds': early_stopping_rounds,
            'random_seed': random_seed,
            'thread_count': thread_count,
            'verbose': 0,
            **kwargs
        }
        self.use_cat_features = use_cat_features
        self.model = None
        self.feature_names_ = None

    def _prepare_data(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if self.feature_names_ is None:
                self.feature_names_ = list(df.columns)
            else:
                df.columns = self.feature_names_
        else:
            if self.feature_names_ is None:
                self.feature_names_ = [f"f_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self.feature_names_)
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(np.int32)
        return df

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor(**self.params)
        
        if self.use_cat_features:
            df_train = self._prepare_data(X_train)
            cat_features = [0]
            eval_set = None
            if X_val is not None and y_val is not None:
                df_val = self._prepare_data(X_val)
                eval_set = (df_val, y_val)
        else:
            df_train = X_train
            cat_features = None
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)

        self.model.fit(
            df_train, y_train,
            eval_set=eval_set,
            cat_features=cat_features,
            verbose=False
        )
        return self

    def predict(self, X):
        if self.use_cat_features:
            if self.feature_names_ is None and hasattr(self.model, 'feature_names_'):
                self.feature_names_ = self.model.feature_names_
            data = self._prepare_data(X)
        else:
            data = X
        preds = self.model.predict(data)
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
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor()
        self.model.load_model(filepath)
        self.feature_names_ = getattr(self.model, 'feature_names_', None)
        return self
