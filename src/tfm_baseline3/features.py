"""Local edge features identical to the GraphSAGE decoder inputs."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPOCH = pd.Timestamp("2000-01-01", tz="UTC")
CATEGORICAL_FEATURES = ["dep_type"]
NUMERIC_FEATURES = [
    "va_num_prev_releases",
    "va_size_bytes",
    "va_num_deps",
    "vb_num_prev_releases",
    "vb_size_bytes",
    "vb_num_deps",
    "release_delta_days",
    "va_release_day",
    "vb_release_day",
    "va_major",
    "va_minor",
    "va_patch",
    "vb_major",
    "vb_minor",
    "vb_patch",
    "specifier_count",
    "has_lower_bound",
    "has_upper_bound",
    "lower_major",
    "lower_minor",
    "lower_patch",
    "upper_major",
    "upper_minor",
    "upper_patch",
    "allows_prereleases",
    "vb_in_range",
]


def raw_edge_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES[:6]].copy()
    va_time = pd.to_datetime(data["va_upload_time"], utc=True)
    vb_time = pd.to_datetime(data["vb_upload_time"], utc=True)
    features["release_delta_days"] = (va_time - vb_time).dt.total_seconds() / 86_400
    features["va_release_day"] = (va_time - EPOCH).dt.total_seconds() / 86_400
    features["vb_release_day"] = (vb_time - EPOCH).dt.total_seconds() / 86_400
    for prefix, values in (("va", data["va"]), ("vb", data["vb"])):
        components = np.asarray([_version_components(value) for value in values])
        features[f"{prefix}_major"] = components[:, 0]
        features[f"{prefix}_minor"] = components[:, 1]
        features[f"{prefix}_patch"] = components[:, 2]
    specifiers = pd.DataFrame(
        [_specifier_features(value) for value in data["declared_range"]], index=data.index
    )
    features = pd.concat([features, specifiers], axis=1)
    features["vb_in_range"] = data["vb_in_range"].astype(float)
    for column in ("va_size_bytes", "vb_size_bytes"):
        features[column] = np.log1p(pd.to_numeric(features[column], errors="coerce").clip(lower=0))
    return features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def build_transformer() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _version_components(value: object) -> tuple[int, int, int]:
    try:
        release = Version(str(value)).release
    except InvalidVersion:
        return (-1, -1, -1)
    padded = (*release, 0, 0, 0)
    return int(padded[0]), int(padded[1]), int(padded[2])


def _specifier_features(value: object) -> dict[str, float]:
    result = {
        "specifier_count": 0.0,
        "has_lower_bound": 0.0,
        "has_upper_bound": 0.0,
        "lower_major": -1.0,
        "lower_minor": -1.0,
        "lower_patch": -1.0,
        "upper_major": -1.0,
        "upper_minor": -1.0,
        "upper_patch": -1.0,
        "allows_prereleases": 0.0,
    }
    try:
        specifiers = list(SpecifierSet(str(value or "")))
    except InvalidSpecifier:
        return result
    result["specifier_count"] = float(len(specifiers))
    result["allows_prereleases"] = float(
        any(_safe_version(spec.version).is_prerelease for spec in specifiers)
    )
    lower = _select_bound(specifiers, {">", ">=", "~=", "=="}, max)
    upper = _select_bound(specifiers, {"<", "<=", "~=", "=="}, min)
    for prefix, bound in (("lower", lower), ("upper", upper)):
        if bound is not None:
            result[f"has_{prefix}_bound"] = 1.0
            major, minor, patch = _version_components(bound)
            result[f"{prefix}_major"] = float(major)
            result[f"{prefix}_minor"] = float(minor)
            result[f"{prefix}_patch"] = float(patch)
    return result


def _select_bound(specifiers: Iterable, operators: set[str], selector) -> str | None:
    candidates = [
        spec.version
        for spec in specifiers
        if spec.operator in operators and not re.search(r"[*]", spec.version)
    ]
    return selector(candidates, key=_safe_version) if candidates else None


def _safe_version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion:
        return Version("0")
