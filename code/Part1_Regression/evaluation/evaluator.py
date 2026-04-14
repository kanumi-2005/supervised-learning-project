import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from scipy.stats import ttest_rel, wilcoxon


class Evaluator:
    def __init__(self, n_splits=10, random_state=42, logger=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logger

    # ====================== LOG ======================

    def _log(self, data):
        if self.logger is not None:
            self.logger.info(data)

    def _log_model_info(self, model):
        def log(data):
            self._log(data)

        def log_estimator(est, name=None):
            info = {
                "type": "model",
                "name": name,
                "class": est.__class__.__name__
            }

            if hasattr(est, "get_params"):
                params = est.get_params(deep=False)
                params = {
                    k: v for k, v in params.items()
                    if v is not None and not callable(v)
                }
                info["params"] = params

            log(info)

            if isinstance(est, Pipeline):
                for step_name, step in est.named_steps.items():
                    log_estimator(step, step_name)

            elif isinstance(est, ColumnTransformer):
                for name, trans, cols in est.transformers:
                    log({
                        "type": "column_transform",
                        "name": name,
                        "columns": cols
                    })

                    if trans not in ["drop", "passthrough"]:
                        log_estimator(trans, name)

        log_estimator(model)

    # ====================== METRICS ======================

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "type": "metrics",
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

        self._log(metrics)
        return metrics

    # ====================== TEST ======================

    def evaluate_test(self, model, X_train, y_train, X_test, y_test):
        self._log({"type": "test_start"})

        self._log_model_info(model)

        self._log({
            "type": "split",
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return self.evaluate(y_test, y_pred)

    # ====================== CROSS VALIDATION ======================

    def cross_validate(self, model, X, y):
        self._log({"type": "cv_start"})

        self._log_model_info(model)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        splits = list(kf.split(X))

        # log splits
        for i, (train_idx, val_idx) in enumerate(splits):
            self._log({
                "type": "fold_split",
                "fold": i,
                "train_idx_sample": train_idx[:5].tolist(),
                "val_idx_sample": val_idx[:5].tolist()
            })

        scores = cross_validate(
            model,
            X,
            y,
            scoring=[
                "neg_mean_squared_error",
                "neg_root_mean_squared_error",
                "neg_mean_absolute_error",
                "r2"
            ],
            cv=splits,
            n_jobs=-1
        )

        mse_list = -scores["test_neg_mean_squared_error"]
        rmse_list = -scores["test_neg_root_mean_squared_error"]
        mae_list = -scores["test_neg_mean_absolute_error"]
        r2_list = scores["test_r2"]

        # log từng fold
        for i in range(len(mse_list)):
            self._log({
                "type": "fold_metrics",
                "fold": i,
                "MSE": mse_list[i],
                "RMSE": rmse_list[i],
                "MAE": mae_list[i],
                "R2": r2_list[i]
            })

        return {
            "MSE": (np.mean(mse_list), np.std(mse_list)),
            "RMSE": (np.mean(rmse_list), np.std(rmse_list)),
            "MAE": (np.mean(mae_list), np.std(mae_list)),
            "R2": (np.mean(r2_list), np.std(r2_list))
        }

    # ====================== COMPARE CV ======================

    def compare_models_cv(self, models, X, y):
        results = []

        for name, model in models.items():
            self._log({
                "type": "cv_model_start",
                "model": name
            })

            scores = self.cross_validate(model, X, y)

            results.append({
                "Model": name,
                "MSE": f"{scores['MSE'][0]:.4f} ± {scores['MSE'][1]:.4f}",
                "RMSE": f"{scores['RMSE'][0]:.4f} ± {scores['RMSE'][1]:.4f}",
                "MAE": f"{scores['MAE'][0]:.4f} ± {scores['MAE'][1]:.4f}",
                "R2": f"{scores['R2'][0]:.4f} ± {scores['R2'][1]:.4f}"
            })

        return pd.DataFrame(results)

    # ====================== COMPARE TEST ======================

    def compare_models_test(self, models, X_train, y_train, X_test, y_test):
        results = []

        for name, model in models.items():
            self._log({
                "type": "test_model_start",
                "model": name
            })

            metrics = self.evaluate_test(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                "Model": name,
                **metrics
            })

        return pd.DataFrame(results)

    # ====================== RAW CV ======================

    def cross_validate_raw(self, model, X, y,
                           scoring="neg_mean_squared_error"):
        self._log({"type": "raw_cv_start"})

        self._log_model_info(model)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        scores = cross_validate(
            model,
            X,
            y,
            scoring=scoring,
            cv=kf,
            n_jobs=-1
        )

        raw_scores = -scores["test_score"]

        self._log({
            "type": "raw_scores",
            "scores": raw_scores.tolist()
        })

        return raw_scores

    # ====================== STAT TEST ======================

    def statistical_test(self, model_a, model_b, X, y,
                         scoring="neg_mean_squared_error"):
        self._log({
            "type": "stat_test_start"
        })

        scores_a = self.cross_validate_raw(model_a, X, y, scoring)
        scores_b = self.cross_validate_raw(model_b, X, y, scoring)

        t_stat, p_t = ttest_rel(scores_a, scores_b)
        w_stat, p_w = wilcoxon(scores_a, scores_b)

        result = {
            "type": "stat_test",
            "model_a_mean": np.mean(scores_a),
            "model_b_mean": np.mean(scores_b),
            "t_test_p": p_t,
            "wilcoxon_p": p_w
        }

        self._log(result)
        return result

    # ====================== COMPARE STAT ======================

    def compare_models_statistical(self, models, X, y,
                                   scoring="neg_mean_squared_error"):
        names = list(models.keys())
        results = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]

                self._log({
                    "type": "compare_pair",
                    "model_a": name_a,
                    "model_b": name_b
                })

                res = self.statistical_test(
                    models[name_a],
                    models[name_b],
                    X, y,
                    scoring
                )

                results.append({
                    "Model A": name_a,
                    "Model B": name_b,
                    "p-value (t-test)": res["t_test_p"],
                    "p-value (Wilcoxon)": res["wilcoxon_p"]
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    from ..pipeline import get_pipeline
    from ..linear_regression.ols import OLS
    from ..linear_regression.mbgd import MBGD
    from ..linear_regression.wls import WLS
    from ..regularization.lasso_regression import LassoRegression

    d = Dataset()
    d.split()

    model1 = get_pipeline(OLS())
    model2 = get_pipeline(MBGD())
    model3 = get_pipeline(LassoRegression())
    models = {"OLS": model1, "MBGD": model2, "Lasso": model3}

    evaluator = Evaluator()

    result = evaluator.compare_models_cv(models, d.X_train, d.y_train)

    print(result)

    result = evaluator.compare_models_test(models, d.X_train, d.y_train,
                                           d.X_test, d.y_test)
    print(result)

    stat_result = evaluator.compare_models_statistical(
        models,
        d.X_train,
        d.y_train
    )

    print(stat_result)
