import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, root_mean_squared_error, \
     mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from scipy.stats import ttest_rel, wilcoxon


class Evaluator:
    def __init__(self, n_splits=10, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def evaluate_test(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return self.evaluate(y_test, y_pred)

    def cross_validate(self, model, X, y):
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
            cv=KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            ),
            n_jobs=-1
        )

        mse_list = -scores["test_neg_mean_squared_error"]
        rmse_list = -scores["test_neg_root_mean_squared_error"]
        mae_list = -scores["test_neg_mean_absolute_error"]
        r2_list = scores["test_r2"]

        return {
            "MSE": (np.mean(mse_list), np.std(mse_list)),
            "RMSE": (np.mean(rmse_list), np.std(rmse_list)),
            "MAE": (np.mean(mae_list), np.std(mae_list)),
            "R2": (np.mean(r2_list), np.std(r2_list))
        }

    def compare_models_cv(self, models, X, y):
        results = []

        for name, model in models.items():
            scores = self.cross_validate(model, X, y)

            results.append({
                "Model": name,
                "MSE": f"{scores['MSE'][0]:.4f} ± {scores['MSE'][1]:.4f}",
                "RMSE": f"{scores['RMSE'][0]:.4f} ± {scores['RMSE'][1]:.4f}",
                "MAE": f"{scores['MAE'][0]:.4f} ± {scores['MAE'][1]:.4f}",
                "R2": f"{scores['R2'][0]:.4f} ± {scores['R2'][1]:.4f}"
            })

        return pd.DataFrame(results)

    def compare_models_test(self, models, X_train, y_train, X_test, y_test):
        results = []

        for name, model in models.items():
            metrics = self.evaluate_test(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            results.append({
                "Model": name,
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"]
            })

        return pd.DataFrame(results)

    def cross_validate_raw(self, model, X, y,
                           scoring="neg_mean_squared_error"):
        scores = cross_validate(
            model,
            X,
            y,
            scoring=scoring,
            cv=KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            ),
            n_jobs=-1
        )

        return -scores["test_score"]

    def statistical_test(self, model_a, model_b, X, y,
                         scoring="neg_mean_squared_error"):
        scores_a = self.cross_validate_raw(model_a, X, y, scoring)
        scores_b = self.cross_validate_raw(model_b, X, y, scoring)

        t_stat, p_t = ttest_rel(scores_a, scores_b)
        w_stat, p_w = wilcoxon(scores_a, scores_b)

        return {
            "Model A mean": np.mean(scores_a),
            "Model B mean": np.mean(scores_b),
            "t-test p-value": p_t,
            "Wilcoxon p-value": p_w
        }

    def compare_models_statistical(self, models, X, y,
                                   scoring="neg_mean_squared_error"):
        names = list(models.keys())
        results = []

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                name_a, name_b = names[i], names[j]

                res = self.statistical_test(
                    models[name_a],
                    models[name_b],
                    X, y
                )

                results.append({
                    "Model A": name_a,
                    "Model B": name_b,
                    "p-value (t-test)": res["t-test p-value"],
                    "p-value (Wilcoxon)": res["Wilcoxon p-value"]
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
