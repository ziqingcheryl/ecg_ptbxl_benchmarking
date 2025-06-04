import sys
from pathlib import Path
import numpy as np

# Add the project's 'code' directory to the path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from utils.utils import apply_thresholds


def test_apply_thresholds_normal():
    preds = np.array([
        [0.8, 0.4, 0.7],
        [0.5, 0.6, 0.4],
    ])
    thresholds = np.array([0.6, 0.5, 0.6])
    expected = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ])
    result = apply_thresholds(preds, thresholds)
    assert np.array_equal(result, expected)


def test_apply_thresholds_selects_max():
    preds = np.array([[0.1, 0.2, 0.15]])
    thresholds = np.array([0.5, 0.5, 0.5])
    expected = np.array([[0, 1, 0]])
    result = apply_thresholds(preds, thresholds)
    assert np.array_equal(result, expected)
