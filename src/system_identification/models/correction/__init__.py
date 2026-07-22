"""Static longitudinal mean/wingbeat correction model contracts."""

from system_identification.models.correction.bundles import StaticCorrectionBundle
from system_identification.models.correction.features import DesignMatrix, build_mean_design, build_waveform_design
from system_identification.models.correction.prediction import predict_cycle_mean, predict_total, predict_waveform
from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.models.correction.static_models import RidgeDiagnostics, RidgeSolution

__all__ = [
    "DesignMatrix",
    "RidgeDiagnostics",
    "RidgeSolution",
    "StaticCorrectionBundle",
    "StaticCorrectionSpec",
    "build_mean_design",
    "build_waveform_design",
    "predict_cycle_mean",
    "predict_total",
    "predict_waveform",
]
