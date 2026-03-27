import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def missing_values(self, show_percent=True):
        df = self.dataset.df
        missing_count = df.isnull().sum()

        result = pd.DataFrame({
            "feature": df.columns,
            "missing_count": missing_count
        })

        if show_percent:
            result["missing_percent"] = (missing_count / len(df) *
                                         100).round(2)

        result = result.sort_values(by="missing_count", ascending=False)
        return result

    def descriptive_stats(self):
        return self.dataset.df.describe()

    def _target_histplot(self, ax, **kwargs):
        return sns.histplot(
            data=self.dataset.df,
            x=self.dataset.target_name,
            ax=ax,
            **kwargs
        )

    def _target_boxplot(self, ax, **kwargs):
        return sns.boxplot(
            data=self.dataset.df,
            x=self.dataset.target_name,
            ax=ax,
            **kwargs
        )

    def plot_target_distribution(self):
        fig, axes = plt.subplots(1, 2, layout="constrained")
        axes = axes.flatten()

        self._target_histplot(ax=axes[0])
        self._target_boxplot(ax=axes[1])

        axes[0].set_title(f"{self.dataset.target_name} Histogram")
        axes[1].set_title(f"{self.dataset.target_name} Boxplot")
        fig.suptitle("Distribution of Target")
        return fig

    def _pair_scatterplot(self, x, y, ax, **kwargs):
        return sns.scatterplot(
            data=self.dataset.df,
            x=x,
            y=y,
            ax=ax,
            **kwargs,
        )

    def plot_scatter_features_target(self):
        n_plots = self.dataset.n_features
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            layout="constrained",
        )
        axes = axes.flatten()

        for i in range(n_plots):
            self._pair_scatterplot(
                self.dataset.feature_names[i],
                self.dataset.target_name,
                axes[i],
            )
            axes[i].set_title(
                (
                    f"{self.dataset.feature_names[i]} vs "
                    f"{self.dataset.target_name} Scatterplot"
                )
            )

        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Features vs Target Scatter Plots")
        return fig

    def plot_corrrelation_matrix(self, annot=True, cmap="coolwarm"):
        corr = self.dataset.df.corr()
        fig, ax = plt.subplots(layout="constrained")
        sns.heatmap(corr, annot=annot, cmap=cmap, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.suptitle("Correlation Matrix")
        return fig

    def _feature_boxplot(self, x, ax, **kwargs):
        return sns.boxplot(
            data=self.dataset.df,
            x=x,
            ax=ax,
            **kwargs
        )

    def plot_outliers(self):
        n_plots = self.dataset.n_features
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            layout="constrained",
        )
        axes = axes.flatten()

        for i in range(n_plots):
            self._feature_boxplot(
                self.dataset.feature_names[i],
                axes[i]
            )

            axes[i].set_title(f"{self.dataset.feature_names[i]} Boxplot")

        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Outliers Detection")
        return fig

if __name__ == "__main__":
    from ..dataset import CaliforniaHousingDataset as Dataset
    d = Dataset()
    eda = EDA(d)

    print(eda.missing_values())
    print(eda.descriptive_stats())

    eda.plot_target_distribution()
    eda.plot_scatter_features_target()
    eda.plot_corrrelation_matrix()
    eda.plot_outliers()
    plt.show()
