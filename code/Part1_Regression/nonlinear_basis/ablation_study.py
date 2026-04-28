import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .rbf import RBF
from .fourier import FourierBasis
from .polynomial import PolynomialBasis
from dataset import CaliforniaHousingDataset as Dataset
from pipeline import get_pipeline


class AblationStudy:
    def __init__(self):
        # =========================
        # LOAD DATA
        # =========================
        d = Dataset()
        d.split()

        X = d.X_train
        y = d.y_train
        
        # load
        self.X_train, self.y_train = d.X_train, d.y_train
        self.X_val, self.y_val = d.X_val, d.y_val
        self.X_test, self.y_test = d.X_test, d.y_test

        # ravel
        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()

        # scale
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)
        self.results = {}

    # =========================
    # EVALUATION (VAL)
    # =========================
    def evaluate_val(self, Xtr, Xval, name):
        model = get_pipeline(LinearRegression())
        model.fit(Xtr, self.y_train)
        pred = model.predict(Xval)

        mse = np.mean((self.y_val - pred) ** 2)
        self.results[name] = mse

    # =========================
    # INIT BASIS (FULL DATA)
    # =========================
    def init_basis(self):
        self.rbf = RBF(n_centers=20, gamma=0.1).fit(self.X_train)
        self.fourier = FourierBasis(n_terms=5).fit(self.X_train)
        self.poly = PolynomialBasis(degree=3).fit(self.X_train)

    # =========================
    # INDIVIDUAL BASIS
    # =========================
    def run_individual(self):
        self.evaluate_val(
            self.rbf.transform(self.X_train),
            self.rbf.transform(self.X_val),
            "RBF"
        )

        self.evaluate_val(
            self.fourier.transform(self.X_train),
            self.fourier.transform(self.X_val),
            "Fourier"
        )

        self.evaluate_val(
            self.poly.transform(self.X_train),
            self.poly.transform(self.X_val),
            "Polynomial"
        )

    # =========================
    # BASIS REMOVAL
    # =========================
    def run_basis_ablation(self):
        self.evaluate_val(
            np.hstack([
                self.fourier.transform(self.X_train),
                self.poly.transform(self.X_train)
            ]),
            np.hstack([
                self.fourier.transform(self.X_val),
                self.poly.transform(self.X_val)
            ]),
            "NO_RBF"
        )

        self.evaluate_val(
            np.hstack([
                self.rbf.transform(self.X_train),
                self.poly.transform(self.X_train)
            ]),
            np.hstack([
                self.rbf.transform(self.X_val),
                self.poly.transform(self.X_val)
            ]),
            "NO_FOURIER"
        )

        self.evaluate_val(
            np.hstack([
                self.rbf.transform(self.X_train),
                self.fourier.transform(self.X_train)
            ]),
            np.hstack([
                self.rbf.transform(self.X_val),
                self.fourier.transform(self.X_val)
            ]),
            "NO_POLY"
        )

    # =========================
    # 🔥 FEATURE GROUP ABLATION (ĐÚNG CHUẨN)
    # =========================
    def run_feature_group_ablation(self):
        def drop_cols(X, cols):
            return np.delete(X, cols, axis=1)

        # 8 features
        econ = [0, 1]
        density = [2, 3, 4, 5]
        spatial = [6, 7]

        def evaluate_group(cols, name):
            Xtr = drop_cols(self.X_train, cols)
            Xval = drop_cols(self.X_val, cols)

            # 🔥 FIT LẠI TOÀN BỘ BASIS
            rbf = RBF(n_centers=20, gamma=0.1).fit(Xtr)
            fourier = FourierBasis(n_terms=5).fit(Xtr)
            poly = PolynomialBasis(degree=3).fit(Xtr)

            # 🔥 COMBINE GIỐNG FULL
            Xtr_comb = np.hstack([
                rbf.transform(Xtr),
                fourier.transform(Xtr),
                poly.transform(Xtr)
            ])

            Xval_comb = np.hstack([
                rbf.transform(Xval),
                fourier.transform(Xval),
                poly.transform(Xval)
            ])

            self.evaluate_val(Xtr_comb, Xval_comb, name)

        evaluate_group(econ, "NO_ECONOMIC")
        evaluate_group(density, "NO_DENSITY")
        evaluate_group(spatial, "NO_SPATIAL")
    
    def run_individual_feature_ablation(self):
        def drop_cols(X, cols):
            return np.delete(X, cols, axis=1)

        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]

        for i, name in enumerate(feature_names):
            Xtr = drop_cols(self.X_train, [i])
            Xval = drop_cols(self.X_val, [i])

            # 🔥 FIT LẠI TOÀN BỘ BASIS (giống FULL)
            rbf = RBF(n_centers=20, gamma=0.1).fit(Xtr)
            fourier = FourierBasis(n_terms=5).fit(Xtr)
            poly = PolynomialBasis(degree=3).fit(Xtr)

            Xtr_comb = np.hstack([
                rbf.transform(Xtr),
                fourier.transform(Xtr),
                poly.transform(Xtr)
            ])

            Xval_comb = np.hstack([
                rbf.transform(Xval),
                fourier.transform(Xval),
                poly.transform(Xval)
            ])

            self.evaluate_val(Xtr_comb, Xval_comb, f"NO_{name}")

    # =========================
    # FULL MODEL
    # =========================
    def run_full(self):
        self.evaluate_val(
            np.hstack([
                self.rbf.transform(self.X_train),
                self.fourier.transform(self.X_train),
                self.poly.transform(self.X_train)
            ]),
            np.hstack([
                self.rbf.transform(self.X_val),
                self.fourier.transform(self.X_val),
                self.poly.transform(self.X_val)
            ]),
            "FULL"
        )

    # =========================
    # RUN ALL
    # =========================
    def run_all(self):
        self.init_basis()
        self.run_individual()
        self.run_basis_ablation()
        self.run_feature_group_ablation()
        self.run_individual_feature_ablation() 
        self.run_full()
    def rank_features(self):
        full = self.results["FULL"]

        importance = {}
        for k, v in self.results.items():
            if k.startswith("NO_"):
                importance[k] = v - full

        # sort giảm dần
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

        print("\n===== FEATURE RANKING (ΔMSE) =====")
        for k, v in sorted_imp:
            print(f"{k:20s} : {v:.6f}")
    # =========================
    # PRINT
    # =========================
    def print_results(self):
        print("\n===== VALIDATION ABLATION RESULTS =====")
        for k, v in self.results.items():
            print(f"{k:20s} : {v:.6f}")

    # =========================
    # PLOT
    # =========================
    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt

        names = list(self.results.keys())
        values = list(self.results.values())

        # =========================
        # SORT (cho đẹp)
        # =========================
        sorted_pairs = sorted(zip(names, values), key=lambda x: x[1])
        names, values = zip(*sorted_pairs)

        # =========================
        # COLOR MAP
        # =========================
        colors = []
        for n in names:
            if n == "FULL":
                colors.append("#FF6B6B")  # đỏ (model lỗi)
            elif "NO_" in n:
                colors.append("#4D96FF")  # xanh dương (ablation)
            else:
                colors.append("#6BCB77")  # xanh lá (individual)

        # =========================
        # PLOT
        # =========================
        plt.figure(figsize=(8,5),constrained_layout=True)
        bars = plt.bar(names, values, color=colors, edgecolor='black')

        # =========================
        # LABEL TRÊN CỘT
        # =========================
        for bar in bars:
            height = bar.get_height()

            # format số
            if height < 10:
                label = f"{height:.3f}"
            elif height < 1000:
                label = f"{height:.1f}"
            else:
                label = f"{height:.0f}"

            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                label,
                ha='center',
                va='bottom',
                fontsize=9,
                rotation=90
            )

        # =========================
        # STYLE
        # =========================
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylabel("MSE (Validation)", fontsize=12)
        plt.title("Ablation Study – Feature & Basis Importance", fontsize=14, weight='bold')

        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # =========================
        # LOG SCALE (rất quan trọng vì có outlier lớn)
        # =========================
        plt.yscale("log")

        # =========================
        # LEGEND
        # =========================
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color="#6BCB77", label="Individual Models"),
            mpatches.Patch(color="#4D96FF", label="Ablation Models"),
            mpatches.Patch(color="#FF6B6B", label="Full Model")
        ]
        plt.legend(handles=legend_handles)
        plt.show()
    def plot_importance(self):
        full = self.results["FULL"]

        names = []
        values = []

        for k, v in self.results.items():
            if k.startswith("NO_"):
                names.append(k)
                values.append(v - full)

        # sort
        pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
        names, values = zip(*pairs)

        plt.figure(figsize=(10,5))
        plt.bar(names, values)
        plt.xticks(rotation=45)
        plt.ylabel("ΔMSE (Importance)")
        plt.title("Feature Importance Ranking")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()