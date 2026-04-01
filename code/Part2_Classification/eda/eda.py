import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.numeric_features = dataset.feature_names[:10]
        self.wilderness_features = dataset.feature_names[10:14]
        self.soil_features = dataset.feature_names[14:]
        self.categorical_features = dataset.feature_names[10:]

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
        return result.sort_values(by="missing_count", ascending=False)

    def target_distribution(self):
        df = self.dataset.df
        counts = df[self.dataset.target_name].value_counts()
        counts_norm = df[self.dataset.target_name].value_counts(normalize=True)
        return pd.DataFrame({
            'count': counts,
            'percent': (counts_norm*100).round(2)
        }).sort_index()

    def plot_target_distribution(self):
        fig, ax = plt.subplots(layout="constrained")
        sns.countplot(x=self.dataset.target_name, data=self.dataset.df, ax=ax)
        ax.set_title(f"{self.dataset.target_name} Countplot")
        fig.suptitle("Target Distribution (Covertype)")
        return fig

    def plot_numeric_features(self):
        n_feats = len(self.numeric_features)
        group_size = math.ceil(n_feats / 2)
        groups = [
            self.numeric_features[:group_size],
            self.numeric_features[group_size:]
        ]

        figs = []
        for gi, group in enumerate(groups):
            fig, axes = plt.subplots(len(group), 2, layout="constrained")
            for i, feat in enumerate(group):
                sns.boxplot(x=feat, data=self.dataset.df, ax=axes[i,0])
                axes[i,0].set_title(f"{feat} Boxplot")

                sns.histplot(self.dataset.df[feat], bins=30, kde=True,
                             ax=axes[i,1])
                axes[i,1].set_title(f"{feat} Histogram")

            fig.suptitle(f"Numeric Features - Group {gi+1}")
            figs.append(fig)
        return figs

    def plot_numeric_vs_target(self):
        n_feats = len(self.numeric_features)
        group_size = math.ceil(n_feats / 2)
        groups = [
            self.numeric_features[:group_size],
            self.numeric_features[group_size:]
        ]

        figs = []
        for gi, group in enumerate(groups):
            fig, axes = plt.subplots(len(group), 1, layout="constrained")
            for i, feat in enumerate(group):
                sns.boxplot(x=self.dataset.target_name, y=feat,
                            data=self.dataset.df, ax=axes[i])
                axes[i].set_title(f"{feat} vs {self.dataset.target_name}")
            fig.suptitle(f"Numeric Features vs Target - Group {gi+1}")
            figs.append(fig)
        return figs

    def plot_categorical_features(self):
        n_feats = len(self.categorical_features)
        group_size = math.ceil(n_feats / 4)
        figs = []
        for gi in range(4):
            group = self.categorical_features[gi*group_size:(gi+1)*group_size]
            if not group:
                continue
            n_cols = 2
            n_rows = math.ceil(len(group) / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, layout="constrained")
            axes = axes.flatten()
            for i, feat in enumerate(group):
                sns.countplot(x=feat, data=self.dataset.df, ax=axes[i])
                axes[i].set_title(f"{feat} Countplot")
            for j in range(len(group), len(axes)):
                axes[j].set_visible(False)
            fig.suptitle(f"Categorical Features - Group {gi+1}")
            figs.append(fig)
        return figs


if __name__ == "__main__":
    from ..dataset import CovtypeDataset as Dataset
    d = Dataset()
    eda = EDA(d)

    print("=== Missing Values ===")
    print(eda.missing_values())
    print("\n=== Target Distribution ===")
    print(eda.target_distribution())

    eda.plot_target_distribution()
    numeric_figs = eda.plot_numeric_features()
    numeric_target_figs = eda.plot_numeric_vs_target()
    categorical_figs = eda.plot_categorical_features()
    plt.show()
