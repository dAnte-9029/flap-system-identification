"""Serializable behavior specification for C2 static correction candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


MODEL_TYPES = frozenset(
    {
        "raw_prior",
        "gain_bias",
        "physical_component_scale",
        "fixed_prior_mean_wb",
        "shaped_prior_mean_wb",
        "no_prior_mean_wb",
    }
)
FORCE_COMPONENTS = frozenset({"fx", "fz"})
CONDITION_SETS = frozenset({"none", "alpha", "frequency", "alpha_frequency"})
MEAN_WEIGHTINGS = frozenset({"equal_cycle", "equal_log", "equal_date"})
WAVEFORM_WEIGHTINGS = frozenset({"equal_sample", "equal_cycle", "equal_log", "equal_date"})
MEAN_WB_TYPES = frozenset({"fixed_prior_mean_wb", "shaped_prior_mean_wb", "no_prior_mean_wb"})


@dataclass(frozen=True)
class StaticCorrectionSpec:
    """A complete, immutable description of one candidate's behavior."""

    model_type: str
    force_component: str
    harmonic_order: int | None = None
    condition_set: str | None = None
    mean_condition_set: str | None = None
    waveform_condition_set: str | None = None
    mean_prior_retention: float | None = None
    waveform_prior_retention: float | None = None
    ridge_lambda_mean: float = 0.0
    ridge_lambda_waveform: float = 0.0
    mean_weighting: str = "equal_cycle"
    waveform_weighting: str = "equal_sample"
    fit_intercept: bool = True
    physical_component: str | None = None
    coefficient_constraints: Mapping[str, object] | None = None
    schema_version: str = "static_correction_spec_v1"

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")
        if self.force_component not in FORCE_COMPONENTS:
            raise ValueError(f"force_component must be one of {sorted(FORCE_COMPONENTS)}")
        legacy_condition = self.condition_set
        mean_condition = self.mean_condition_set
        waveform_condition = self.waveform_condition_set
        if legacy_condition is not None:
            if legacy_condition not in CONDITION_SETS:
                raise ValueError(f"condition_set must be one of {sorted(CONDITION_SETS)}")
            if mean_condition is not None and mean_condition != legacy_condition:
                raise ValueError("condition_set conflicts with mean_condition_set")
            if waveform_condition is not None and waveform_condition != legacy_condition:
                raise ValueError("condition_set conflicts with waveform_condition_set")
            mean_condition = legacy_condition
            waveform_condition = legacy_condition
        else:
            mean_condition = "none" if mean_condition is None else mean_condition
            waveform_condition = "none" if waveform_condition is None else waveform_condition
        if mean_condition not in CONDITION_SETS:
            raise ValueError(f"mean_condition_set must be one of {sorted(CONDITION_SETS)}")
        if waveform_condition not in CONDITION_SETS:
            raise ValueError(f"waveform_condition_set must be one of {sorted(CONDITION_SETS)}")
        object.__setattr__(self, "mean_condition_set", mean_condition)
        object.__setattr__(self, "waveform_condition_set", waveform_condition)
        if self.mean_weighting not in MEAN_WEIGHTINGS:
            raise ValueError(f"mean_weighting must be one of {sorted(MEAN_WEIGHTINGS)}")
        if self.waveform_weighting not in WAVEFORM_WEIGHTINGS:
            raise ValueError(f"waveform_weighting must be one of {sorted(WAVEFORM_WEIGHTINGS)}")
        for name, value in (
            ("ridge_lambda_mean", self.ridge_lambda_mean),
            ("ridge_lambda_waveform", self.ridge_lambda_waveform),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            ("mean_prior_retention", self.mean_prior_retention),
            ("waveform_prior_retention", self.waveform_prior_retention),
        ):
            if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} retention must be finite and within [0, 1]")

        if self.model_type in MEAN_WB_TYPES:
            if self.harmonic_order not in {1, 2, 3, 4}:
                raise ValueError("mean/WB harmonic_order must be one of 1, 2, 3, 4")
            if not self.fit_intercept:
                raise ValueError("mean/WB candidates require fit_intercept=True")
            if self.physical_component is not None:
                raise ValueError("mean/WB candidates do not accept physical_component")
            if self.model_type == "fixed_prior_mean_wb" and (
                self.mean_prior_retention != 1.0 or self.waveform_prior_retention != 1.0
            ):
                raise ValueError("fixed-prior retention must be exactly 1 for both branches")
            if self.model_type == "no_prior_mean_wb" and (
                self.mean_prior_retention != 0.0 or self.waveform_prior_retention != 0.0
            ):
                raise ValueError("no-prior retention must be exactly 0 for both branches")
            if self.model_type == "shaped_prior_mean_wb" and (
                self.mean_prior_retention is None or self.waveform_prior_retention is None
            ):
                raise ValueError("shaped-prior retention must be explicit for both branches")
        else:
            if self.harmonic_order is not None:
                raise ValueError(f"{self.model_type} does not accept harmonic_order")
            if mean_condition != "none" or waveform_condition != "none":
                raise ValueError(f"{self.model_type} does not accept condition features")
            if self.mean_prior_retention is not None or self.waveform_prior_retention is not None:
                raise ValueError(f"{self.model_type} does not accept mean/WB retention")

        if self.model_type == "raw_prior":
            if self.ridge_lambda_mean != 0.0 or self.ridge_lambda_waveform != 0.0:
                raise ValueError("raw_prior does not fit ridge coefficients")
            if self.fit_intercept:
                raise ValueError("raw_prior requires fit_intercept=False because it fits no coefficients")
            if self.mean_weighting != "equal_cycle" or self.waveform_weighting != "equal_sample":
                raise ValueError("raw_prior requires canonical no-fit weighting placeholders")
            if self.physical_component is not None or self.coefficient_constraints is not None:
                raise ValueError("raw_prior does not accept component fields")
        elif self.model_type == "gain_bias":
            if self.ridge_lambda_mean != 0.0:
                raise ValueError("gain_bias uses ridge_lambda_waveform only")
            if self.physical_component is not None or self.coefficient_constraints is not None:
                raise ValueError("gain_bias does not accept component fields")
            if not self.fit_intercept:
                raise ValueError("gain_bias requires fit_intercept=True")
        elif self.model_type == "physical_component_scale":
            if self.force_component != "fz":
                raise ValueError("normal-force physical_component_scale currently supports fz only")
            if self.physical_component != "normal_force":
                raise ValueError("physical_component_scale requires physical_component='normal_force'")
            if self.fit_intercept:
                raise ValueError("physical_component_scale requires fit_intercept=False")
            if self.ridge_lambda_mean != 0.0:
                raise ValueError("physical_component_scale uses ridge_lambda_waveform only")
            constraints = dict(self.coefficient_constraints or {})
            if constraints:
                if constraints.get("strategy", "clip_after_fit") != "clip_after_fit":
                    raise ValueError("Only clip_after_fit component constraint strategy is supported")
                minimum = float(constraints.get("scale_min", 0.0))
                maximum = float(constraints.get("scale_max", 2.0))
                if not (math.isfinite(minimum) and math.isfinite(maximum) and minimum <= maximum):
                    raise ValueError("Invalid component scale constraint bounds")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.coefficient_constraints is not None:
            result["coefficient_constraints"] = dict(self.coefficient_constraints)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StaticCorrectionSpec":
        return cls(**dict(value))


@dataclass(frozen=True)
class StaticModelFamilyConfig:
    schema_version: str
    force_components: tuple[str, ...]
    model_types: tuple[str, ...]
    harmonic_orders: tuple[int, ...]
    condition_sets: tuple[str, ...]
    prior_retention_values: tuple[float, ...]
    ridge_values_for_future_c3: tuple[float, ...]
    allowed_fit_partitions: tuple[str, ...]
    forbidden_features: tuple[str, ...]
    authority: Mapping[str, object]
    smoke_defaults: Mapping[str, object]


def _tuple(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def parse_model_family_config(value: Mapping[str, object]) -> StaticModelFamilyConfig:
    """Validate candidate-space configuration without enumerating or selecting it."""

    schema_version = str(value.get("schema_version", ""))
    if schema_version != "static_mean_wb_family_v1":
        raise ValueError(f"Unsupported static model family schema_version: {schema_version!r}")
    force_components = tuple(str(item) for item in _tuple(value.get("force_components"), "force_components"))
    model_types = tuple(str(item) for item in _tuple(value.get("model_types"), "model_types"))
    harmonic_orders = tuple(int(item) for item in _tuple(value.get("harmonic_orders"), "harmonic_orders"))
    condition_sets = tuple(str(item) for item in _tuple(value.get("condition_sets"), "condition_sets"))
    retentions = tuple(float(item) for item in _tuple(value.get("prior_retention_values"), "prior_retention_values"))
    ridges = tuple(float(item) for item in _tuple(value.get("ridge_values_for_future_c3"), "ridge_values_for_future_c3"))
    allowed = tuple(str(item) for item in _tuple(value.get("allowed_fit_partitions"), "allowed_fit_partitions"))
    forbidden = tuple(str(item) for item in _tuple(value.get("forbidden_features"), "forbidden_features"))
    if set(force_components) != FORCE_COMPONENTS:
        raise ValueError("force_components must contain exactly fx and fz")
    if set(model_types) != MODEL_TYPES:
        raise ValueError("model_types must contain the complete C2 candidate family")
    if set(harmonic_orders) != {1, 2, 3, 4}:
        raise ValueError("harmonic_orders must contain exactly K=1..4")
    if set(condition_sets) != CONDITION_SETS:
        raise ValueError("condition_sets must contain the four C2 condition contracts")
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in retentions):
        raise ValueError("prior_retention_values must be finite and within [0, 1]")
    if set(retentions) != {0.0, 0.25, 0.5, 0.75, 1.0}:
        raise ValueError("prior_retention_values must contain the fixed C2 discrete set")
    if any(not math.isfinite(item) or item < 0.0 for item in ridges):
        raise ValueError("ridge_values_for_future_c3 must be finite and non-negative")
    if allowed != ("train",):
        raise ValueError("C2 allowed_fit_partitions must be exactly ['train']")
    required_forbidden = {"airspeed", "dynamic_pressure", "history", "future_state"}
    if not required_forbidden.issubset(forbidden):
        raise ValueError("forbidden_features omits a required C2 prohibition")
    authority = value.get("authority")
    smoke_defaults = value.get("smoke_defaults")
    if not isinstance(authority, Mapping) or not isinstance(smoke_defaults, Mapping):
        raise ValueError("authority and smoke_defaults must be mappings")
    return StaticModelFamilyConfig(
        schema_version=schema_version,
        force_components=force_components,
        model_types=model_types,
        harmonic_orders=harmonic_orders,
        condition_sets=condition_sets,
        prior_retention_values=retentions,
        ridge_values_for_future_c3=ridges,
        allowed_fit_partitions=allowed,
        forbidden_features=forbidden,
        authority=dict(authority),
        smoke_defaults=dict(smoke_defaults),
    )
