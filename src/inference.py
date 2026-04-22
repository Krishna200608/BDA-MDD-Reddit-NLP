from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.exceptions import InconsistentVersionWarning

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional at import time
    torch = None
    AutoModel = None
    AutoTokenizer = None


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "results_summary.csv"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"

DEFAULT_LABEL_ORDER = ["Control", "Moderate MDD", "Severe Ideation"]
USER_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


@dataclass
class PredictionResult:
    model_key: str
    model_display_name: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    runtime_notes: list[str]
    explanation_rows: list[dict[str, Any]]
    benchmark_row: dict[str, Any] | None


def load_json_file(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def get_label_order() -> list[str]:
    class_payload = load_json_file(CLASS_LABELS_PATH, {"label_order": DEFAULT_LABEL_ORDER})
    label_order = class_payload.get("label_order", DEFAULT_LABEL_ORDER)
    return [str(label) for label in label_order]


def get_model_metadata() -> dict[str, Any]:
    fallback = {
        "default_model_key": "tfidf_logreg",
        "models": {
            "tfidf_logreg": {
                "display_name": "TF-IDF + Logistic Regression",
                "artifact_path": "models/tfidf_logreg_pipeline.joblib",
                "type": "sparse_pipeline",
                "supports_probabilities": True,
                "is_available": False,
            },
            "tfidf_linearsvc": {
                "display_name": "TF-IDF + LinearSVC",
                "artifact_path": "models/tfidf_linearsvc_pipeline.joblib",
                "type": "sparse_pipeline",
                "supports_probabilities": False,
                "is_available": False,
            },
            "roberta_rf": {
                "display_name": "TwitterRoBERTa + Random Forest",
                "artifact_path": "models/roberta_rf_classifier.joblib",
                "type": "roberta_rf",
                "encoder_name": "cardiffnlp/twitter-roberta-base",
                "supports_probabilities": True,
                "is_available": False,
            },
        },
        "dashboard": {
            "title": "Reddit MDD NLP Dashboard",
            "subtitle": "Three-Class Mental Health Language Analysis",
            "system_badges": ["System Ready", "Colab-Trained", "Live Inference"],
        },
    }
    metadata = load_json_file(MODEL_METADATA_PATH, fallback)
    metadata.setdefault("models", fallback["models"])
    metadata.setdefault("default_model_key", fallback["default_model_key"])
    metadata.setdefault("dashboard", fallback["dashboard"])
    return metadata


def get_available_models() -> dict[str, dict[str, Any]]:
    metadata = get_model_metadata()
    models = metadata.get("models", {})
    for model_key, info in models.items():
        artifact_path = REPO_ROOT / info.get("artifact_path", "")
        info["resolved_artifact_path"] = str(artifact_path)
        info["is_available"] = artifact_path.exists()
    return models


def get_default_model_key() -> str:
    metadata = get_model_metadata()
    available_models = get_available_models()
    requested = metadata.get("default_model_key", "tfidf_logreg")
    if available_models.get(requested, {}).get("is_available"):
        return requested
    for model_key, model_info in available_models.items():
        if model_info.get("is_available"):
            return model_key
    return requested


@lru_cache(maxsize=8)
def load_saved_artifact(path_str: str):
    return joblib.load(path_str)


@lru_cache(maxsize=2)
def load_roberta_components(encoder_name: str):
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise RuntimeError("Transformers/PyTorch dependencies are unavailable for the RoBERTa inference path.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()
    return tokenizer, model, device


def preprocess_for_roberta(text: str) -> str:
    normalized = str(text or "").strip()
    normalized = USER_PATTERN.sub("@user", normalized)
    normalized = URL_PATTERN.sub("http", normalized)
    return normalized


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denom = exp_values.sum()
    if denom == 0:
        return np.full_like(exp_values, 1 / len(exp_values), dtype=float)
    return exp_values / denom


def normalize_probabilities(probabilities: np.ndarray, label_order: list[str]) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    total = clipped.sum()
    if total <= 0:
        clipped = np.ones(len(label_order), dtype=float) / len(label_order)
    else:
        clipped = clipped / total
    return {label: float(score) for label, score in zip(label_order, clipped)}


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def preprocess_for_sparse(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = URL_PATTERN.sub("", normalized)
    normalized = normalized.replace("\n", " ")
    normalized = NON_ALPHA_PATTERN.sub("", normalized)
    normalized = MULTISPACE_PATTERN.sub(" ", normalized).strip()
    filtered_tokens = [token for token in normalized.split() if token not in get_stop_words()]
    return " ".join(filtered_tokens)


def get_results_summary_rows() -> list[dict[str, Any]]:
    if not RESULTS_SUMMARY_PATH.exists():
        return []
    import pandas as pd

    df = pd.read_csv(RESULTS_SUMMARY_PATH)
    return df.to_dict(orient="records")


def lookup_benchmark_row(model_display_name: str) -> dict[str, Any] | None:
    for row in get_results_summary_rows():
        if row.get("model") == model_display_name and row.get("evaluation_type") == "holdout":
            return row
    return None


def explain_sparse_prediction(pipeline, text: str, predicted_index: int, label_order: list[str], top_n: int = 8) -> list[dict[str, Any]]:
    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    vector = tfidf.transform([text])
    feature_names = np.asarray(tfidf.get_feature_names_out())
    non_zero_indices = vector.nonzero()[1]
    if len(non_zero_indices) == 0:
        return []

    coef_matrix = getattr(classifier, "coef_", None)
    if coef_matrix is None:
        return []

    class_coefficients = coef_matrix[predicted_index]
    contributions = vector[:, non_zero_indices].toarray().ravel() * class_coefficients[non_zero_indices]

    rows = []
    for feature_idx, contribution in zip(non_zero_indices, contributions):
        rows.append(
            {
                "token": str(feature_names[feature_idx]),
                "contribution": float(contribution),
                "direction": "supports prediction" if contribution >= 0 else "pushes away",
                "predicted_label": label_order[predicted_index],
            }
        )

    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return rows[:top_n]


def get_roberta_embedding(text: str, encoder_name: str) -> np.ndarray:
    tokenizer, model, device = load_roberta_components(encoder_name)
    processed_text = preprocess_for_roberta(text)
    inputs = tokenizer(
        [processed_text],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
    return cls_embedding


def predict_with_sparse_pipeline(model_info: dict[str, Any], text: str, label_order: list[str]) -> PredictionResult:
    artifact = load_saved_artifact(model_info["resolved_artifact_path"])
    model_display_name = str(model_info["display_name"])
    model_input_text = preprocess_for_sparse(text)
    predicted_index = int(artifact.predict([model_input_text])[0])

    if hasattr(artifact, "predict_proba"):
        probability_vector = artifact.predict_proba([model_input_text])[0]
    else:
        decision_vector = artifact.decision_function([model_input_text])[0]
        if np.ndim(decision_vector) == 0:
            decision_vector = np.asarray([0.0, float(decision_vector)])
        probability_vector = softmax(np.asarray(decision_vector, dtype=float))

    probabilities = normalize_probabilities(probability_vector, label_order)
    predicted_label = label_order[predicted_index]
    confidence = probabilities[predicted_label]

    return PredictionResult(
        model_key="",
        model_display_name=model_display_name,
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
        cleaned_text=model_input_text,
        runtime_notes=[
            "Sparse inference path loaded from saved sklearn pipeline.",
            "Input text was normalized with the same regex + stopword cleaning strategy used to build selftext_cleaned.",
        ],
        explanation_rows=explain_sparse_prediction(artifact, model_input_text, predicted_index, label_order),
        benchmark_row=lookup_benchmark_row(model_display_name),
    )


def predict_with_roberta_rf(model_info: dict[str, Any], text: str, label_order: list[str]) -> PredictionResult:
    classifier = load_saved_artifact(model_info["resolved_artifact_path"])
    encoder_name = str(model_info.get("encoder_name", "cardiffnlp/twitter-roberta-base"))
    processed_text = preprocess_for_roberta(text)
    embedding = get_roberta_embedding(processed_text, encoder_name)
    predicted_index = int(classifier.predict(embedding)[0])

    if hasattr(classifier, "predict_proba"):
        probability_vector = classifier.predict_proba(embedding)[0]
    else:
        raw_prediction = classifier.predict(embedding)[0]
        probability_vector = np.eye(len(label_order))[int(raw_prediction)]

    probabilities = normalize_probabilities(probability_vector, label_order)
    predicted_label = label_order[predicted_index]
    confidence = probabilities[predicted_label]

    return PredictionResult(
        model_key="",
        model_display_name=str(model_info["display_name"]),
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
        cleaned_text=processed_text,
        runtime_notes=[
            f"Encoder loaded at runtime: {encoder_name}",
            "Model-card preprocessing applied: usernames -> @user, links -> http",
        ],
        explanation_rows=[],
        benchmark_row=lookup_benchmark_row(str(model_info["display_name"])),
    )


def predict_text(model_key: str, text: str) -> PredictionResult:
    label_order = get_label_order()
    models = get_available_models()
    if model_key not in models:
        raise KeyError(f"Unknown model key: {model_key}")

    model_info = models[model_key]
    if not model_info.get("is_available"):
        raise FileNotFoundError(
            f"Saved artifact not found for {model_key}. Expected: {model_info.get('resolved_artifact_path')}"
        )

    if model_info.get("type") == "roberta_rf":
        result = predict_with_roberta_rf(model_info, text, label_order)
    else:
        result = predict_with_sparse_pipeline(model_info, text, label_order)

    result.model_key = model_key
    return result


def label_theme(label: str) -> dict[str, str]:
    themes = {
        "Control": {"accent": "#0f766e", "soft": "#d1fae5", "border": "#99f6e4", "badge": "#14b8a6"},
        "Moderate MDD": {"accent": "#c2410c", "soft": "#ffedd5", "border": "#fdba74", "badge": "#f59e0b"},
        "Severe Ideation": {"accent": "#b91c1c", "soft": "#fee2e2", "border": "#fca5a5", "badge": "#ef4444"},
    }
    return themes.get(label, {"accent": "#1d4ed8", "soft": "#dbeafe", "border": "#93c5fd", "badge": "#2563eb"})


def summarize_input_text(text: str) -> dict[str, int]:
    stripped = str(text).strip()
    words = [token for token in stripped.split() if token]
    return {
        "characters": len(stripped),
        "words": len(words),
        "lines": stripped.count("\n") + (1 if stripped else 0),
    }


def get_sample_inputs() -> dict[str, str]:
    return {
        "Neutral conversation": (
            "I had a long chat with friends today, made coffee, and spent the evening talking about random life goals."
        ),
        "Moderate distress": (
            "I feel exhausted and numb lately. I still go through the routine, but everything feels heavy and I keep losing motivation."
        ),
        "High-risk language": (
            "I feel trapped and hopeless, and lately I keep thinking that everyone would be better off without me."
        ),
    }


def get_dashboard_summary() -> dict[str, Any]:
    metadata = get_model_metadata()
    dashboard = metadata.get("dashboard", {})
    return {
        "title": dashboard.get("title", "Reddit MDD NLP Dashboard"),
        "subtitle": dashboard.get("subtitle", "Three-Class Mental Health Language Analysis"),
        "system_badges": dashboard.get("system_badges", ["System Ready", "Live Inference"]),
    }
