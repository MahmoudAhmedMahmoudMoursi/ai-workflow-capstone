import logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def get_logger(log_file: str = "aavail_monitoring.log", logger_name: str = "aavail_monitor") -> logging.Logger:
    """Create a logger isolated to the specified log file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "").endswith(log_file)
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def detect_drift(reference_series: pd.Series, current_series: pd.Series, alpha: float = 0.05) -> dict:
    """Detect distribution drift/novelty using the Kolmogorov-Smirnov test."""
    ref_clean = pd.to_numeric(reference_series, errors="coerce").dropna()
    cur_clean = pd.to_numeric(current_series, errors="coerce").dropna()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return {
            "drift_detected": False,
            "p_value": None,
            "statistic": None,
            "reason": "Insufficient data for drift test",
        }

    statistic, p_value = ks_2samp(ref_clean, cur_clean)
    drift_detected = bool(p_value < alpha)

    return {
        "drift_detected": drift_detected,
        "p_value": float(p_value),
        "statistic": float(statistic),
        "alpha": alpha,
    }


def log_prediction_event(
    country: str,
    target_date: str,
    prediction: float,
    drift_info: dict = None,
    log_file: str = "aavail_monitoring.log",
) -> None:
    """Log prediction details and optional drift status."""
    logger = get_logger(log_file=log_file)
    logger.info(
        f"country={country} | target_date={target_date} | prediction={prediction:.4f} | drift={drift_info}"
    )
