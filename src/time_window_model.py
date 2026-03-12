from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    ColumnTransformer = None
    ExtraTreesRegressor = None
    RandomForestRegressor = None
    SimpleImputer = None
    LinearRegression = None
    Pipeline = None
    SKLEARN_AVAILABLE = False

from src.config import AppConfig
from src.io_utils import write_dataframe
from src.tle_features import build_feature_table


LOGGER = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "mean_motion",
    "delta_mean_motion",
    "bstar",
    "delta_bstar",
    "eccentricity",
    "inclination",
    "semi_major_axis_km",
    "mean_motion_slope_per_hour",
    "bstar_slope_per_hour",
    "semi_major_axis_slope_km_per_hour",
    "delta_hours",
    "epoch_dayofyear",
    "epoch_hour",
]


@dataclass(slots=True)
class TimeWindowResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    selected_model_name: str


def _build_dataset(gp_histories: dict[int, pd.DataFrame], selected_cases: pd.DataFrame) -> pd.DataFrame:
    case_lookup = selected_cases.set_index("norad_id").to_dict(orient="index")
    frames: list[pd.DataFrame] = []
    for norad_id, gp_history_df in gp_histories.items():
        case = case_lookup.get(int(norad_id))
        if not case:
            continue
        reentry_time = pd.to_datetime(case["reentry_time_utc"], utc=True, errors="coerce")
        if pd.isna(reentry_time):
            continue
        features = build_feature_table(gp_history_df, reentry_time_utc=reentry_time)
        if features.empty:
            continue
        features["norad_id"] = int(norad_id)
        features["object_name"] = case.get("object_name")
        features["reentry_time_utc"] = reentry_time
        frames.append(features)
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=["hours_to_decay", "mean_motion"])
    return dataset


def _build_splits(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in dataset.groupby("norad_id"):
        group = group.sort_values("epoch").reset_index(drop=True)
        if len(group) < 3:
            train_parts.append(group)
            continue
        test_count = 2 if len(group) >= 6 else 1
        train_parts.append(group.iloc[:-test_count])
        test_parts.append(group.iloc[-test_count:])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=dataset.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=dataset.columns)
    return train_df, test_df


def _heuristic_predict(frame: pd.DataFrame) -> pd.Series:
    slope = frame["mean_motion_slope_per_hour"].fillna(frame["delta_mean_motion"] / frame["delta_hours"].replace(0, np.nan))
    slope = slope.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    terminal_mean_motion = max(frame["mean_motion"].max() + 0.05, 16.0)
    predicted = (terminal_mean_motion - frame["mean_motion"]) / slope.replace(0.0, np.nan)
    predicted = predicted.replace([np.inf, -np.inf], np.nan).abs()
    fallback = frame["hours_to_decay"].median() if frame["hours_to_decay"].notna().any() else 24.0
    return predicted.fillna(fallback).clip(lower=0.0, upper=30.0 * 24.0)


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def _fit_and_score(train_df: pd.DataFrame, test_df: pd.DataFrame, config: AppConfig) -> tuple[dict[str, object], pd.DataFrame]:
    models: dict[str, object] = {}
    metrics_rows: list[dict[str, float | str]] = []
    if train_df.empty or test_df.empty or not SKLEARN_AVAILABLE:
        return models, pd.DataFrame(metrics_rows)

    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), FEATURE_COLUMNS)],
        remainder="drop",
    )
    candidate_models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=config.random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=config.random_state,
            n_jobs=1,
        ),
    }

    for model_name, model in candidate_models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["hours_to_decay"])
        predictions = pipeline.predict(test_df[FEATURE_COLUMNS])
        mae = _mae(test_df["hours_to_decay"], predictions)
        rmse = _rmse(test_df["hours_to_decay"], predictions)
        metrics_rows.append({"model_name": model_name, "mae_hours": mae, "rmse_hours": rmse})
        models[model_name] = pipeline
    return models, pd.DataFrame(metrics_rows)


def _extract_feature_importance(model_name: str, fitted_pipeline: object) -> pd.DataFrame:
    model_step = fitted_pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importance_values = getattr(model_step, "feature_importances_")
    elif hasattr(model_step, "coef_"):
        importance_values = np.abs(np.ravel(getattr(model_step, "coef_")))
    else:
        return pd.DataFrame(columns=["feature", "importance", "model_name"])

    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance_values, "model_name": model_name})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_time_window_model(
    gp_histories: dict[int, pd.DataFrame],
    selected_cases: pd.DataFrame,
    config: AppConfig,
) -> TimeWindowResult | None:
    dataset = _build_dataset(gp_histories, selected_cases)
    if dataset.empty:
        LOGGER.info("Skipping time-window model because no GP history features were available.")
        return None

    train_df, test_df = _build_splits(dataset)
    predictions_frames: list[pd.DataFrame] = []
    feature_importance = pd.DataFrame(columns=["feature", "importance"])
    metrics = pd.DataFrame()
    selected_model_name = "heuristic"

    if len(train_df) >= config.min_training_rows and not test_df.empty and SKLEARN_AVAILABLE:
        models, metrics = _fit_and_score(train_df, test_df, config)
        if not metrics.empty:
            selected_model_name = metrics.sort_values("mae_hours").iloc[0]["model_name"]
            best_model = models[selected_model_name]
            test_predictions = test_df.copy()
            test_predictions["predicted_hours_to_decay"] = best_model.predict(test_df[FEATURE_COLUMNS])
            test_predictions["model_name"] = selected_model_name
            test_predictions["split"] = "test"
            predictions_frames.append(test_predictions)

            feature_importance = _extract_feature_importance(selected_model_name, best_model)

            latest_rows = dataset.sort_values("epoch").groupby("norad_id").tail(1).copy()
            latest_rows["predicted_hours_to_decay"] = best_model.predict(latest_rows[FEATURE_COLUMNS])
            latest_rows["model_name"] = selected_model_name
            latest_rows["split"] = "latest_row"
            predictions_frames.append(latest_rows)

    if not predictions_frames:
        heuristic_rows = dataset.sort_values("epoch").groupby("norad_id").tail(3).copy()
        heuristic_rows["predicted_hours_to_decay"] = _heuristic_predict(heuristic_rows)
        heuristic_rows["model_name"] = "heuristic"
        heuristic_rows["split"] = "latest_rows"
        predictions_frames.append(heuristic_rows)
        metrics = pd.DataFrame(
            [
                {
                    "model_name": "heuristic",
                    "mae_hours": _mae(heuristic_rows["hours_to_decay"], heuristic_rows["predicted_hours_to_decay"]),
                    "rmse_hours": _rmse(heuristic_rows["hours_to_decay"], heuristic_rows["predicted_hours_to_decay"]),
                }
            ]
        )

    predictions = pd.concat(predictions_frames, ignore_index=True)
    predictions = predictions[
        [
            "norad_id",
            "object_name",
            "epoch",
            "reentry_time_utc",
            "hours_to_decay",
            "predicted_hours_to_decay",
            "model_name",
            "split",
        ]
    ].rename(columns={"hours_to_decay": "actual_hours_to_decay"})

    write_dataframe(predictions, config.outputs_tables_dir / "time_window_predictions.csv")
    if not metrics.empty:
        write_dataframe(metrics, config.outputs_tables_dir / "time_window_model_metrics.csv")
    if not feature_importance.empty:
        write_dataframe(feature_importance, config.outputs_tables_dir / "time_window_feature_importance.csv")

    return TimeWindowResult(
        predictions=predictions,
        metrics=metrics,
        feature_importance=feature_importance,
        selected_model_name=selected_model_name,
    )
