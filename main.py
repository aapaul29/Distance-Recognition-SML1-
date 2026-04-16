"""
SML Project 1 — Obstacle Distance Estimation (Improved Pipeline)

Pipeline:
  1. Load RGB images at downsample_factor=5  (10,800 raw features)
  2. Feature engineering: pixel features + HOG descriptors
  3. StandardScaler → PCA (200 components, ~98-99% variance)
  4. Multiple diverse models:
     - HistGradientBoostingRegressor (tuned via RandomizedSearchCV)
     - ExtraTreesRegressor
     - KNeighborsRegressor
     - MLPRegressor
     - Ridge (baseline)
  5. Stacking with diverse base learners → Ridge meta-learner
  6. Final evaluation + Kaggle submission

Grading thresholds: MAE ≤ 28 cm (1.0) | ≤ 18 cm (4.0) | ≤ 8 cm (6.0)

NOTE: SVRs are NOT allowed in this project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ── sklearn imports ──────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    StackingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── scipy imports (for RandomizedSearchCV distributions) ─────────────────────
from scipy.stats import uniform, randint

# ── skimage imports (for HOG feature engineering) ────────────────────────────
from skimage.feature import hog

# ── project utilities ────────────────────────────────────────────────────────
from utils import load_config, load_dataset, load_test_dataset, print_results, save_results


# =============================================================================
#  FEATURE ENGINEERING
# =============================================================================

def extract_hog_features(images_flat, img_height, img_width, n_channels):
    """
    Extract HOG (Histogram of Oriented Gradients) features from flattened images.
    HOG captures edge / gradient structure which strongly correlates with depth.
    """
    hog_features = []
    for i in range(len(images_flat)):
        img = images_flat[i].reshape(img_height, img_width, n_channels) if n_channels > 1 \
              else images_flat[i].reshape(img_height, img_width)

        # Use channel_axis for multichannel images
        if n_channels > 1:
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=-1,
                feature_vector=True,
            )
        else:
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                feature_vector=True,
            )
        hog_features.append(features)

    return np.array(hog_features)


def extract_spatial_features(images_flat, img_height, img_width, n_channels):
    """
    Extract spatial statistics per image quadrant.
    Divides image into 4 quadrants and computes mean, std, min, max per quadrant.
    """
    total_pixels = img_height * img_width * n_channels
    features = []
    for i in range(len(images_flat)):
        img = images_flat[i].reshape(img_height, img_width, n_channels) if n_channels > 1 \
              else images_flat[i].reshape(img_height, img_width)

        mid_h = img_height // 2
        mid_w = img_width // 2

        quadrants = [
            img[:mid_h, :mid_w],   # top-left
            img[:mid_h, mid_w:],   # top-right
            img[mid_h:, :mid_w],   # bottom-left
            img[mid_h:, mid_w:],   # bottom-right
        ]

        row = []
        for q in quadrants:
            row.extend([q.mean(), q.std(), q.min(), q.max()])

        # Also add full-image stats
        row.extend([img.mean(), img.std(), img.min(), img.max()])
        # Add pixel intensity variance (useful depth cue)
        row.append(np.var(images_flat[i]))

        features.append(row)

    return np.array(features)


def build_features(images_flat, config):
    """
    Combine raw pixel features with engineered features.
    Returns the combined feature matrix.
    """
    IMAGE_SIZE = (300, 300)
    img_height = IMAGE_SIZE[0] // config["downsample_factor"]
    img_width  = IMAGE_SIZE[1] // config["downsample_factor"]
    n_channels = 3 if config["load_rgb"] else 1

    print(f"[INFO]: Image dimensions: {img_height}x{img_width}x{n_channels}")
    print(f"[INFO]: Raw pixel features: {images_flat.shape[1]}")

    # 1. HOG features
    print("[INFO]: Extracting HOG features...")
    t0 = time.time()
    hog_feats = extract_hog_features(images_flat, img_height, img_width, n_channels)
    print(f"[INFO]: HOG features: {hog_feats.shape[1]} (took {time.time()-t0:.1f}s)")

    # 2. Spatial statistics
    print("[INFO]: Extracting spatial statistics...")
    spatial_feats = extract_spatial_features(images_flat, img_height, img_width, n_channels)
    print(f"[INFO]: Spatial features: {spatial_feats.shape[1]}")

    # 3. Combine all features
    combined = np.hstack([images_flat, hog_feats, spatial_feats])
    print(f"[INFO]: Combined features: {combined.shape[1]}")

    return combined


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    # ── Load configs from "config.yaml" ──────────────────────────────────────
    config = load_config()

    # ── Load dataset ─────────────────────────────────────────────────────────
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    print(f"[INFO]: Feature dimensionality: {images.shape[1]}")
    print(f"[INFO]: Distance range: {distances.min():.2f} m – {distances.max():.2f} m")

    # ── Feature Engineering ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FEATURE ENGINEERING")
    print("="*60)
    X_all = build_features(images, config)

    # ── Train / Validation / Test Split ──────────────────────────────────────
    # train (70%) | val (15%) | test (15%)
    print("\n" + "="*60)
    print("  CREATING SPLITS")
    print("="*60)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, distances, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42
    )
    print(f"[INFO]: Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── Preprocessing: StandardScaler + PCA ──────────────────────────────────
    print("\n" + "="*60)
    print("  PREPROCESSING (Scaler + PCA)")
    print("="*60)

    N_COMPONENTS = 200  # Increased from 100 — captures more variance with richer features

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"[INFO]: PCA with {N_COMPONENTS} components explains {explained:.1f}% of variance")
    print(f"[INFO]: Reduced feature shape: {X_train_pca.shape}")

    # ── Baseline: Ridge Regression ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL TRAINING")
    print("="*60)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_pca, y_train)
    val_pred_ridge = ridge.predict(X_val_pca)
    mae_ridge = mean_absolute_error(y_val, val_pred_ridge) * 100
    print(f"Ridge (baseline)        — Val MAE: {mae_ridge:.2f} cm")

    # ── HistGradientBoostingRegressor (tuned via RandomizedSearchCV) ──────────
    print("\n[INFO]: Running RandomizedSearchCV for HGBR (this may take a while)...")

    param_dist = {
        'learning_rate': uniform(0.01, 0.19),       # uniform between 0.01 and 0.20
        'max_depth':     randint(3, 15),             # integers 3..14
        'max_iter':      randint(200, 2001),         # integers 200..2000
        'min_samples_leaf': randint(1, 30),          # integers 1..29
        'l2_regularization': uniform(0.0, 1.0),      # uniform between 0 and 1
        'max_leaf_nodes': randint(20, 100),           # integers 20..99
    }

    random_search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_dist,
        n_iter=80,
        scoring='neg_mean_absolute_error',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    random_search.fit(X_train_pca, y_train)

    best_hgbr = random_search.best_estimator_
    print(f"Best HGBR params: {random_search.best_params_}")

    val_pred_hgbr = best_hgbr.predict(X_val_pca)
    mae_hgbr = mean_absolute_error(y_val, val_pred_hgbr) * 100
    r2_hgbr  = r2_score(y_val, val_pred_hgbr) * 100
    print(f"HGBR (tuned)            — Val MAE: {mae_hgbr:.2f} cm  |  R²: {r2_hgbr:.1f}%")

    # ── ExtraTreesRegressor ──────────────────────────────────────────────────
    etr = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    etr.fit(X_train_pca, y_train)
    val_pred_etr = etr.predict(X_val_pca)
    mae_etr = mean_absolute_error(y_val, val_pred_etr) * 100
    r2_etr  = r2_score(y_val, val_pred_etr) * 100
    print(f"ExtraTrees              — Val MAE: {mae_etr:.2f} cm  |  R²: {r2_etr:.1f}%")

    # ── KNeighborsRegressor ──────────────────────────────────────────────────
    # After PCA, KNN in low-dimensional space is very effective for this task
    best_knn_mae = float('inf')
    best_k = 5
    for k in [3, 5, 7, 10, 15, 20]:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
        knn.fit(X_train_pca, y_train)
        pred = knn.predict(X_val_pca)
        mae = mean_absolute_error(y_val, pred) * 100
        if mae < best_knn_mae:
            best_knn_mae = mae
            best_k = k

    knn_model = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
    knn_model.fit(X_train_pca, y_train)
    val_pred_knn = knn_model.predict(X_val_pca)
    mae_knn = mean_absolute_error(y_val, val_pred_knn) * 100
    r2_knn  = r2_score(y_val, val_pred_knn) * 100
    print(f"KNN (k={best_k}, weighted)  — Val MAE: {mae_knn:.2f} cm  |  R²: {r2_knn:.1f}%")

    # ── MLPRegressor (Neural Network) ────────────────────────────────────────
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    mlp.fit(X_train_pca, y_train)
    val_pred_mlp = mlp.predict(X_val_pca)
    mae_mlp = mean_absolute_error(y_val, val_pred_mlp) * 100
    r2_mlp  = r2_score(y_val, val_pred_mlp) * 100
    print(f"MLP (256-128-64)        — Val MAE: {mae_mlp:.2f} cm  |  R²: {r2_mlp:.1f}%")

    # ── AdaBoostRegressor ────────────────────────────────────────────────────
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=6),
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )
    ada.fit(X_train_pca, y_train)
    val_pred_ada = ada.predict(X_val_pca)
    mae_ada = mean_absolute_error(y_val, val_pred_ada) * 100
    r2_ada  = r2_score(y_val, val_pred_ada) * 100
    print(f"AdaBoost                — Val MAE: {mae_ada:.2f} cm  |  R²: {r2_ada:.1f}%")

    # ── Diverse Stacking ─────────────────────────────────────────────────────
    # Key: use DIVERSE base learners (tree-based + distance-based + linear + neural)
    print("\n[INFO]: Training stacking ensemble (diverse base learners)...")

    stack_estimators = [
        ('hgbr', HistGradientBoostingRegressor(
            **random_search.best_params_, random_state=42)),
        ('etr', ExtraTreesRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42)),
        ('knn', KNeighborsRegressor(
            n_neighbors=best_k, weights='distance', n_jobs=-1)),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, random_state=42)),
        ('ridge', Ridge(alpha=10.0)),
    ]

    stack = StackingRegressor(
        estimators=stack_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )
    stack.fit(X_train_pca, y_train)

    val_pred_stack = stack.predict(X_val_pca)
    mae_stack = mean_absolute_error(y_val, val_pred_stack) * 100
    r2_stack  = r2_score(y_val, val_pred_stack) * 100
    print(f"Stacking (diverse)      — Val MAE: {mae_stack:.2f} cm  |  R²: {r2_stack:.1f}%")

    # ── Model Comparison ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)

    results = {
        'Ridge (baseline)':    mae_ridge,
        'HGBR (tuned)':        mae_hgbr,
        'ExtraTrees':          mae_etr,
        f'KNN (k={best_k})':   mae_knn,
        'MLP':                 mae_mlp,
        'AdaBoost':            mae_ada,
        'Stacking (diverse)':  mae_stack,
    }

    # Sort by MAE (best first)
    results = dict(sorted(results.items(), key=lambda x: x[1]))

    for name, mae in results.items():
        marker = "✓" if mae <= 18 else ("~" if mae <= 28 else "✗")
        print(f"  [{marker}] {name:30s} — MAE: {mae:.2f} cm")

    print(f"\n  Thresholds: ≤28 cm (1.0) | ≤18 cm (4.0) | ≤8 cm (6.0)")

    # ── Pick the best model ──────────────────────────────────────────────────
    best_name = min(results, key=results.get)
    best_mae  = results[best_name]
    print(f"\n  → Best model: {best_name} (MAE: {best_mae:.2f} cm)")

    # Map names to actual model objects
    model_map = {
        'Ridge (baseline)':    ridge,
        'HGBR (tuned)':        best_hgbr,
        'ExtraTrees':          etr,
        f'KNN (k={best_k})':   knn_model,
        'MLP':                 mlp,
        'AdaBoost':            ada,
        'Stacking (diverse)':  stack,
    }
    final_model = model_map[best_name]

    # ── Final Test Set Evaluation ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL TEST SET RESULTS")
    print("="*60)

    test_pred = final_model.predict(X_test_pca)
    print_results(y_test, test_pred)

    # ── Retrain on full labelled dataset & generate Kaggle submission ────────
    print("\n" + "="*60)
    print("  KAGGLE SUBMISSION")
    print("="*60)

    # Rebuild features for ALL labelled data
    X_all_features = build_features(images, config)

    X_all_scaled = scaler.fit_transform(X_all_features)
    X_all_pca    = pca.fit_transform(X_all_scaled)

    final_model.fit(X_all_pca, distances)
    print("[INFO]: Final model retrained on full dataset")

    # Load and transform Kaggle test images
    kaggle_images_raw = load_test_dataset(config)
    kaggle_images_raw = np.array(kaggle_images_raw)

    kaggle_features = build_features(kaggle_images_raw, config)
    kaggle_scaled   = scaler.transform(kaggle_features)
    kaggle_pca      = pca.transform(kaggle_scaled)

    kaggle_pred = final_model.predict(kaggle_pca)

    save_results(kaggle_pred)
    print(f"[INFO]: Saved prediction.csv with {len(kaggle_pred)} predictions")
    print(f"[INFO]: Predicted distance range: {kaggle_pred.min():.2f} m – {kaggle_pred.max():.2f} m")

    # ── Visualization ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Model comparison bar chart
    ax = axes[0]
    names = list(results.keys())
    maes  = list(results.values())
    colors = ['#2ecc71' if m <= 8 else '#f39c12' if m <= 18 else '#e74c3c' for m in maes]
    bars = ax.barh(names, maes, color=colors)
    ax.axvline(28, color='red',    linestyle='--', alpha=0.7, label='Grade 1.0 (28 cm)')
    ax.axvline(18, color='orange', linestyle='--', alpha=0.7, label='Grade 4.0 (18 cm)')
    ax.axvline(8,  color='green',  linestyle='--', alpha=0.7, label='Grade 6.0 (8 cm)')
    ax.set_xlabel('Val MAE (cm)')
    ax.set_title('Model Comparison')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, maes):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=8)

    # Predicted vs actual scatter
    ax2 = axes[1]
    ax2.scatter(y_test * 100, test_pred * 100, alpha=0.4, s=10, color='steelblue')
    lims = [min((y_test * 100).min(), (test_pred * 100).min()),
            max((y_test * 100).max(), (test_pred * 100).max())]
    ax2.plot(lims, lims, 'r--', alpha=0.7, label='Perfect prediction')
    ax2.set_xlabel('Actual Distance (cm)')
    ax2.set_ylabel('Predicted Distance (cm)')
    ax2.set_title(f'Test Set: Predicted vs Actual (MAE={mean_absolute_error(y_test, test_pred)*100:.1f} cm)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()
    print("[INFO]: Saved model_comparison.png")