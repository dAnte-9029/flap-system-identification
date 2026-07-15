"""Pure NumPy DeLaurier (1993) strip loads and strip-integrated wrench.

The equations and component reference points are synchronized from
``dAnte-9029/IsaacLab`` commit
``3b5d4ec1d28f1384cf042402992ad7ea59995f49``. The module is intentionally
independent of IsaacLab, Isaac Sim, PhysX, and Torch.

Internal Wang axes are right-handed: ``x`` is root-to-tip and the pitching
axis, ``y`` is surface-normal, and ``z`` points chordwise toward the leading
edge. Forces are polar vectors. Moments are axial vectors. Strip moments are
about the wing-root pitching-axis origin unless explicitly translated.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DeLaurierParams:
    alpha0_rad: float = math.radians(0.5)
    eta_s: float = 0.98
    cd_cf: float = 1.98
    alpha_stall_min_rad: float = -1.0e9
    alpha_stall_max_rad: float = math.radians(13.0)
    xi: float = 0.0
    c_mac: float = 0.025
    nu: float = 1.5e-5
    cd_f: float | None = None
    stall_smoothing_width_rad: float = 0.0


@dataclass(frozen=True)
class WingGeometry:
    """One-wing strip geometry; all arrays have shape ``(N,)`` in metres."""

    x_mid: np.ndarray
    dx: np.ndarray
    chord: np.ndarray
    d_hat: np.ndarray
    semi_span_m: float
    area_m2: float
    aspect_ratio: float


@dataclass(frozen=True)
class DeLaurierStripLoads:
    """Raw component loads shaped ``(B,N)`` in N or N m."""

    dN_c: np.ndarray
    dN_a: np.ndarray
    dT_s: np.ndarray
    dD_camber: np.ndarray
    dD_f: np.ndarray
    dM_ac: np.ndarray
    dM_a: np.ndarray
    span: np.ndarray
    chord: np.ndarray
    strip_width: np.ndarray
    d_hat: np.ndarray
    resultant_normal_force: np.ndarray
    resultant_chordwise_force: np.ndarray
    power_input: np.ndarray
    separation_weight: np.ndarray
    alpha: np.ndarray
    alpha_prime: np.ndarray
    alpha_le: np.ndarray
    alpha_stall: np.ndarray
    reduced_frequency: np.ndarray

    @property
    def normal_force_total(self) -> np.ndarray:
        return self.dN_c + self.dN_a

    @property
    def chordwise_force_total(self) -> np.ndarray:
        return self.dT_s - self.dD_camber - self.dD_f


@dataclass(frozen=True)
class DeLaurierStripWrench:
    """Integrated one-wing Wang-frame wrench about the wing-root origin."""

    force_wang: np.ndarray
    moment_wang_about_wing_origin: np.ndarray
    force_normal_wang: np.ndarray
    force_chordwise_wang: np.ndarray
    force_from_dN_c_wang: np.ndarray
    force_from_dN_a_wang: np.ndarray
    force_from_dT_s_wang: np.ndarray
    force_from_dD_camber_wang: np.ndarray
    force_from_dD_f_wang: np.ndarray
    moment_from_dN_c_wang: np.ndarray
    moment_from_dN_a_wang: np.ndarray
    moment_from_dT_s_wang: np.ndarray
    moment_from_dD_camber_wang: np.ndarray
    moment_from_dD_f_wang: np.ndarray
    moment_from_dM_ac_wang: np.ndarray
    moment_from_dM_a_wang: np.ndarray

    @property
    def moment_from_force_wang(self) -> np.ndarray:
        return (
            self.moment_from_dN_c_wang
            + self.moment_from_dN_a_wang
            + self.moment_from_dT_s_wang
            + self.moment_from_dD_camber_wang
            + self.moment_from_dD_f_wang
        )

    @property
    def moment_from_free_couple_wang(self) -> np.ndarray:
        return self.moment_from_dM_ac_wang + self.moment_from_dM_a_wang


def load_wing_geometry_csv(path: str | Path, *, num_strips: int = 80, d_hat: float = 0.0) -> WingGeometry:
    """Reproduce the frozen CSV-to-uniform-strip geometry builder."""

    source = Path(path)
    x_values: list[float] = []
    chord_values: list[float] = []
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {"x_mid_m", "c_m", "dhat"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Invalid wing geometry CSV header in {source}")
        for row in reader:
            x_values.append(float(row["x_mid_m"]))
            chord_values.append(float(row["c_m"]))
    order = np.argsort(x_values)
    x_csv = np.asarray(x_values, dtype=float)[order]
    chord_csv = np.asarray(chord_values, dtype=float)[order]
    if len(x_csv) < 2 or np.any(np.diff(x_csv) <= 0.0):
        raise ValueError("Wing geometry CSV requires at least two increasing strip centers")
    csv_dx = float(np.median(np.diff(x_csv)))
    semi_span = float(x_csv[-1] + 0.5 * csv_dx)
    csv_area = float(np.sum(chord_csv * csv_dx))
    aspect_ratio = semi_span * semi_span / csv_area
    if int(num_strips) <= 0:
        raise ValueError("num_strips must be positive")
    dx = np.full(int(num_strips), semi_span / int(num_strips), dtype=float)
    x_mid = (np.arange(int(num_strips), dtype=float) + 0.5) * dx
    chord = np.interp(np.clip(x_mid, x_csv[0], x_csv[-1]), x_csv, chord_csv)
    d_hat_values = np.full(int(num_strips), float(d_hat), dtype=float)
    return WingGeometry(
        x_mid=x_mid,
        dx=dx,
        chord=chord,
        d_hat=d_hat_values,
        semi_span_m=semi_span,
        area_m2=float(np.sum(chord * dx)),
        aspect_ratio=float(aspect_ratio),
    )


def _broadcast_strip(values: np.ndarray | float, *, batch: int, strips: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.broadcast_to(array, (batch, strips))
    if array.ndim == 1:
        if array.shape[0] == batch:
            return np.broadcast_to(array[:, None], (batch, strips))
        if array.shape[0] == strips:
            return np.broadcast_to(array[None, :], (batch, strips))
    if array.ndim == 2:
        try:
            return np.broadcast_to(array, (batch, strips))
        except ValueError as exc:
            raise ValueError(f"Cannot broadcast {name} from {array.shape} to {(batch, strips)}") from exc
    raise ValueError(f"Invalid {name} shape {array.shape}")


def _f_g_prime(k: np.ndarray, aspect_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    c1 = (0.5 * aspect_ratio) / (2.32 + aspect_ratio)
    c2 = 0.181 + 0.772 / aspect_ratio
    k2 = np.square(k)
    denominator = k2 + c2 * c2
    return 1.0 - c1 * k2 / denominator, -(c1 * c2 * k) / denominator


def _cd_f_from_reynolds(reynolds: np.ndarray) -> np.ndarray:
    log_re = np.log10(np.maximum(reynolds, 2.0))
    return 0.89 / np.maximum(log_re, 1.0e-3) ** 2.58


def compute_delaurier_strip_loads(
    h: np.ndarray,
    hdot: np.ndarray,
    hddot: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    theta_ddot: np.ndarray,
    wing_geometry: WingGeometry,
    rho: np.ndarray | float,
    airspeed_m_s: np.ndarray | float,
    *,
    theta_a: np.ndarray | float,
    theta_bar: np.ndarray | float,
    omega_ref_rad_s: np.ndarray | float,
    params: DeLaurierParams,
    enable_separation: bool = False,
) -> DeLaurierStripLoads:
    """Compute frozen DeLaurier strip components without span integration."""

    h_input = np.asarray(h, dtype=float)
    if h_input.ndim != 2:
        raise ValueError("h must have shape (B,N)")
    batch, strips = h_input.shape
    if strips != len(wing_geometry.x_mid):
        raise ValueError("h strip count does not match geometry")
    h = _broadcast_strip(h, batch=batch, strips=strips, name="h")
    hdot = _broadcast_strip(hdot, batch=batch, strips=strips, name="hdot")
    hddot = _broadcast_strip(hddot, batch=batch, strips=strips, name="hddot")
    theta = _broadcast_strip(theta, batch=batch, strips=strips, name="theta")
    theta_dot = _broadcast_strip(theta_dot, batch=batch, strips=strips, name="theta_dot")
    theta_ddot = _broadcast_strip(theta_ddot, batch=batch, strips=strips, name="theta_ddot")
    span = np.broadcast_to(wing_geometry.x_mid, (batch, strips))
    width = np.broadcast_to(wing_geometry.dx, (batch, strips))
    chord = np.broadcast_to(wing_geometry.chord, (batch, strips))
    d_hat = np.broadcast_to(wing_geometry.d_hat, (batch, strips))
    rho = _broadcast_strip(rho, batch=batch, strips=strips, name="rho")
    airspeed = _broadcast_strip(airspeed_m_s, batch=batch, strips=strips, name="airspeed_m_s")
    airspeed_safe = np.maximum(airspeed, 1.0e-6)
    theta_a = _broadcast_strip(theta_a, batch=batch, strips=strips, name="theta_a")
    theta_bar = _broadcast_strip(theta_bar, batch=batch, strips=strips, name="theta_bar")
    omega_ref = _broadcast_strip(
        omega_ref_rad_s,
        batch=batch,
        strips=strips,
        name="omega_ref_rad_s",
    )

    reduced_frequency = chord * omega_ref / (2.0 * airspeed_safe)
    f_prime, g_prime = _f_g_prime(reduced_frequency, wing_geometry.aspect_ratio)
    g_over_k = np.divide(
        g_prime,
        reduced_frequency,
        out=np.zeros_like(g_prime),
        where=np.abs(reduced_frequency) > 1.0e-8,
    )
    theta_minus = theta - theta_a
    alpha = (
        hdot * np.cos(theta_minus)
        + 0.75 * chord * theta_dot
        + airspeed * (theta - theta_bar)
    ) / airspeed_safe
    alpha_dot = (
        hddot * np.cos(theta_minus)
        - hdot * np.sin(theta_minus) * theta_dot
        + 0.75 * chord * theta_ddot
        + airspeed * theta_dot
    ) / airspeed_safe
    w0_over_u = 2.0 * (float(params.alpha0_rad) + theta_bar) / (2.0 + wing_geometry.aspect_ratio)
    alpha_prime = (wing_geometry.aspect_ratio / (2.0 + wing_geometry.aspect_ratio)) * (
        f_prime * alpha + chord / (2.0 * airspeed_safe) * g_over_k * alpha_dot
    ) - w0_over_u

    vx = airspeed * np.cos(theta) - hdot * np.sin(theta_minus)
    vy = airspeed * (alpha_prime + theta_bar) - 0.5 * chord * theta_dot
    velocity = np.sqrt(np.square(vx) + np.square(vy))
    cn = 2.0 * np.pi * (alpha_prime + float(params.alpha0_rad) + theta_bar)
    qn = 0.5 * rho * airspeed * velocity
    dN_c = qn * cn * chord * width
    v2_dot = airspeed * alpha_dot - 0.25 * chord * theta_ddot
    dN_a = rho * np.pi * np.square(chord) / 4.0 * v2_dot * width
    dN_attached = dN_c + dN_a
    dD_camber = -2.0 * np.pi * float(params.alpha0_rad) * (alpha_prime + theta_bar) * qn * chord * width
    dT_s = (
        float(params.eta_s)
        * 2.0
        * np.pi
        * np.square(alpha_prime + theta_bar - 0.25 * chord * theta_dot / airspeed_safe)
        * qn
        * chord
        * width
    )
    if params.cd_f is None:
        cd_f = _cd_f_from_reynolds(airspeed_safe * chord / max(float(params.nu), 1.0e-9))
    else:
        cd_f = np.full_like(chord, float(params.cd_f))
    dD_f = cd_f * (0.5 * rho * np.square(vx)) * chord * width
    dF_x_attached = dT_s - dD_camber - dD_f

    alpha_stall = float(params.alpha_stall_max_rad) + float(params.xi) * np.sqrt(
        np.maximum(chord * np.abs(alpha_dot) / (2.0 * airspeed_safe), 0.0)
    )
    alpha_le = alpha_prime + theta_bar - 0.75 * chord * theta_dot / airspeed_safe
    attached = (alpha_le >= float(params.alpha_stall_min_rad)) & (alpha_le <= alpha_stall)
    vn = hdot * np.cos(theta_minus) + 0.5 * chord * theta_dot + airspeed * np.sin(theta)
    vhat = np.sqrt(np.square(vx) + np.square(vn))
    dN_separated = float(params.cd_cf) * (0.5 * rho * vhat * vn) * chord * width + 0.5 * dN_a
    smoothing = float(params.stall_smoothing_width_rad)
    if enable_separation and smoothing > 0.0:
        w_hi = 1.0 / (1.0 + np.exp(-(alpha_le - alpha_stall) / smoothing))
        w_lo = 1.0 / (1.0 + np.exp(-(float(params.alpha_stall_min_rad) - alpha_le) / smoothing))
        separation_weight = np.clip(w_lo + w_hi - w_lo * w_hi, 0.0, 1.0)
        resultant_normal = (1.0 - separation_weight) * dN_attached + separation_weight * dN_separated
        resultant_chordwise = (1.0 - separation_weight) * dF_x_attached
    elif enable_separation:
        separation_weight = np.where(attached, 0.0, 1.0)
        resultant_normal = np.where(attached, dN_attached, dN_separated)
        resultant_chordwise = np.where(attached, dF_x_attached, 0.0)
    else:
        separation_weight = np.zeros_like(dN_attached)
        resultant_normal = dN_attached
        resultant_chordwise = dF_x_attached

    dM_ac = float(params.c_mac) * (0.5 * rho * airspeed * velocity) * np.square(chord) * width
    dM_a = -(
        rho * np.pi * chord**3 * (theta_dot * airspeed) / 16.0
        + rho * np.pi * chord**4 * theta_ddot / 128.0
    ) * width
    dP_attached = (
        resultant_chordwise * hdot * np.sin(theta_minus)
        + resultant_normal * (hdot * np.cos(theta_minus) + 0.25 * chord * theta_dot)
        + dN_a * (0.25 * chord * theta_dot)
        - dM_ac * theta_dot
        - dM_a * theta_dot
    )
    dP_separated = dN_separated * (hdot * np.cos(theta_minus) + 0.5 * chord * theta_dot)
    if enable_separation and smoothing > 0.0:
        power = (1.0 - separation_weight) * dP_attached + separation_weight * dP_separated
    elif enable_separation:
        power = np.where(attached, dP_attached, dP_separated)
    else:
        power = dP_attached
    return DeLaurierStripLoads(
        dN_c=dN_c,
        dN_a=dN_a,
        dT_s=dT_s,
        dD_camber=dD_camber,
        dD_f=dD_f,
        dM_ac=dM_ac,
        dM_a=dM_a,
        span=span,
        chord=chord,
        strip_width=width,
        d_hat=d_hat,
        resultant_normal_force=resultant_normal,
        resultant_chordwise_force=resultant_chordwise,
        power_input=power,
        separation_weight=separation_weight,
        alpha=alpha,
        alpha_prime=alpha_prime,
        alpha_le=alpha_le,
        alpha_stall=alpha_stall,
        reduced_frequency=reduced_frequency,
    )


def _force_vector(*, normal: np.ndarray | None = None, chordwise: np.ndarray | None = None) -> np.ndarray:
    scalar = normal if normal is not None else chordwise
    if scalar is None:
        raise ValueError("normal or chordwise force is required")
    zeros = np.zeros_like(scalar)
    return np.stack((zeros, normal if normal is not None else zeros, chordwise if chordwise is not None else zeros), axis=-1)


def integrate_delaurier_strip_wrench(
    loads: DeLaurierStripLoads,
    *,
    include_aerodynamic_center_moment: bool = True,
    include_apparent_mass_moment: bool = True,
) -> DeLaurierStripWrench:
    """Integrate attached components about the wing-root pitching-axis origin."""

    zeros = np.zeros_like(loads.span)
    chordwise_point = np.stack((loads.span, zeros, zeros), axis=-1)
    normal_circulatory_point = np.stack(
        (loads.span, zeros, (loads.d_hat - 0.25) * loads.chord),
        axis=-1,
    )
    normal_apparent_point = np.stack(
        (loads.span, zeros, (loads.d_hat - 0.50) * loads.chord),
        axis=-1,
    )
    force_dN_c = _force_vector(normal=loads.dN_c)
    force_dN_a = _force_vector(normal=loads.dN_a)
    force_dT_s = _force_vector(chordwise=loads.dT_s)
    force_dD_camber = _force_vector(chordwise=-loads.dD_camber)
    force_dD_f = _force_vector(chordwise=-loads.dD_f)

    def integrated_moment(point: np.ndarray, force: np.ndarray) -> np.ndarray:
        return np.sum(np.cross(point, force), axis=1)

    moment_dN_c = integrated_moment(normal_circulatory_point, force_dN_c)
    moment_dN_a = integrated_moment(normal_apparent_point, force_dN_a)
    moment_dT_s = integrated_moment(chordwise_point, force_dT_s)
    moment_dD_camber = integrated_moment(chordwise_point, force_dD_camber)
    moment_dD_f = integrated_moment(chordwise_point, force_dD_f)
    zeros_batch = np.zeros(loads.dM_ac.shape[0], dtype=float)
    moment_dM_ac = np.stack((np.sum(loads.dM_ac, axis=1), zeros_batch, zeros_batch), axis=-1)
    moment_dM_a = np.stack((np.sum(loads.dM_a, axis=1), zeros_batch, zeros_batch), axis=-1)
    if not include_aerodynamic_center_moment:
        moment_dM_ac.fill(0.0)
    if not include_apparent_mass_moment:
        moment_dM_a.fill(0.0)

    force_components = [force_dN_c, force_dN_a, force_dT_s, force_dD_camber, force_dD_f]
    integrated_forces = [np.sum(component, axis=1) for component in force_components]
    force_wang = sum(integrated_forces)
    moment_wang = sum(
        [moment_dN_c, moment_dN_a, moment_dT_s, moment_dD_camber, moment_dD_f, moment_dM_ac, moment_dM_a]
    )
    return DeLaurierStripWrench(
        force_wang=force_wang,
        moment_wang_about_wing_origin=moment_wang,
        force_normal_wang=integrated_forces[0] + integrated_forces[1],
        force_chordwise_wang=integrated_forces[2] + integrated_forces[3] + integrated_forces[4],
        force_from_dN_c_wang=integrated_forces[0],
        force_from_dN_a_wang=integrated_forces[1],
        force_from_dT_s_wang=integrated_forces[2],
        force_from_dD_camber_wang=integrated_forces[3],
        force_from_dD_f_wang=integrated_forces[4],
        moment_from_dN_c_wang=moment_dN_c,
        moment_from_dN_a_wang=moment_dN_a,
        moment_from_dT_s_wang=moment_dT_s,
        moment_from_dD_camber_wang=moment_dD_camber,
        moment_from_dD_f_wang=moment_dD_f,
        moment_from_dM_ac_wang=moment_dM_ac,
        moment_from_dM_a_wang=moment_dM_a,
    )


def transform_wrench(
    force: np.ndarray,
    moment: np.ndarray,
    polar_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform polar force and axial moment, including reflection parity."""

    force_values = np.asarray(force, dtype=float)
    moment_values = np.asarray(moment, dtype=float)
    transform = np.asarray(polar_transform, dtype=float)
    transformed_force = np.einsum("...ij,...j->...i", transform, force_values)
    polar_moment = np.einsum("...ij,...j->...i", transform, moment_values)
    return transformed_force, np.linalg.det(transform)[..., None] * polar_moment


def translate_wrench_moment(
    force: np.ndarray,
    moment_about_origin: np.ndarray,
    origin_position: np.ndarray,
    reference_position: np.ndarray,
) -> np.ndarray:
    """Return ``M_G=M_O+(p_O-p_G) cross F`` in one proper Cartesian frame."""

    return np.asarray(moment_about_origin, dtype=float) + np.cross(
        np.asarray(origin_position, dtype=float) - np.asarray(reference_position, dtype=float),
        np.asarray(force, dtype=float),
    )
