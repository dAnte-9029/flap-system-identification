/****************************************************************************
 *
 *   Copyright (c) 2019-2023 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file rate_control.cpp
 */

#include "rate_control.hpp"
#include <px4_platform_common/defines.h>

using namespace matrix;

RateControl::RollILimits RateControl::computeRollILimits(float imax_raw, float transferred_i_raw,
		float headroom_release_ratio)
{
	RollILimits limits{};

	if (!PX4_ISFINITE(imax_raw) || imax_raw < 0.f || !PX4_ISFINITE(transferred_i_raw)
	    || !PX4_ISFINITE(headroom_release_ratio) || headroom_release_ratio < 0.f
	    || headroom_release_ratio > 1.f) {
		return limits;
	}

	limits.total_limit = imax_raw + headroom_release_ratio * fabsf(transferred_i_raw);
	limits.lower = math::max(-imax_raw, -limits.total_limit - transferred_i_raw);
	limits.upper = math::min(imax_raw, limits.total_limit - transferred_i_raw);
	limits.valid = PX4_ISFINITE(limits.total_limit) && PX4_ISFINITE(limits.lower)
		       && PX4_ISFINITE(limits.upper) && limits.lower <= limits.upper;
	return limits;
}

RateControl::RollITransferResult RateControl::applyRollITransfer(const RollITransferRequest &request)
{
	RollITransferResult result{};
	result.requested_delta_raw = request.requested_delta_raw;
	result.residual_before_raw = _rate_int(0);
	result.residual_after_raw = _rate_int(0);
	result.reset_epoch = _roll_i_reset_epoch;

	const float limit = _lim_int(0);
	const RollILimits current_limits = computeRollILimits(limit, request.transferred_i_raw,
			request.headroom_release_ratio);
	const float total_before = _rate_int(0) + request.transferred_i_raw;
	constexpr float StateTolerance = 4.f * FLT_EPSILON;

	if (!PX4_ISFINITE(request.requested_delta_raw) || !PX4_ISFINITE(request.transferred_i_raw)
	    || !PX4_ISFINITE(_rate_int(0)) || !current_limits.valid
	    || _rate_int(0) < current_limits.lower - StateTolerance
	    || _rate_int(0) > current_limits.upper + StateTolerance
	    || !PX4_ISFINITE(total_before) || fabsf(total_before) > current_limits.total_limit + StateTolerance) {
		result.reason = !PX4_ISFINITE(total_before) || (current_limits.valid
				&& fabsf(total_before) > current_limits.total_limit + StateTolerance)
				? RollITransferLimitReason::TotalEquivalentLimit : RollITransferLimitReason::Invalid;
		return result;
	}

	float accepted_delta = request.requested_delta_raw;

	if (request.mode == RollITransferMode::TowardZero) {
		if (fabsf(_rate_int(0)) <= FLT_EPSILON && fabsf(accepted_delta) > FLT_EPSILON) {
			accepted_delta = 0.f;
			result.limited_at_zero = true;
			result.reason = RollITransferLimitReason::ZeroCrossing;
		}

		const bool wrong_direction = (_rate_int(0) > 0.f && accepted_delta > 0.f)
					     || (_rate_int(0) < 0.f && accepted_delta < 0.f);

		if (wrong_direction) {
			result.reason = RollITransferLimitReason::WrongDirection;
			return result;
		}

		if (fabsf(accepted_delta) > fabsf(_rate_int(0))) {
			accepted_delta = -_rate_int(0);
			result.limited_at_zero = true;
			result.reason = RollITransferLimitReason::ZeroCrossing;
		}
	}

	const float unconstrained = _rate_int(0) + accepted_delta;

	if (!PX4_ISFINITE(unconstrained)) {
		result.reason = RollITransferLimitReason::Invalid;
		return result;
	}

	const float transferred_after = request.validate_post_state ? request.transferred_i_after_raw
					: request.transferred_i_raw;
	const RollILimits post_limits = computeRollILimits(limit, transferred_after,
			request.headroom_release_ratio);

	if (!post_limits.valid) {
		result.reason = RollITransferLimitReason::Invalid;
		return result;
	}

	const float constrained = math::constrain(unconstrained, post_limits.lower, post_limits.upper);
	result.limited_by_imax = constrained < unconstrained || constrained > unconstrained;

	if (result.limited_by_imax) {
		result.reason = fabsf(unconstrained) > limit ? RollITransferLimitReason::IntegratorLimit
				: RollITransferLimitReason::TotalEquivalentLimit;
	}

	_rate_int(0) = constrained;
	result.accepted_delta_raw = constrained - result.residual_before_raw;
	result.residual_after_raw = constrained;
	result.valid = true;
	return result;
}

void RateControl::setPidGains(const Vector3f &P, const Vector3f &I, const Vector3f &D)
{
	_gain_p = P;
	_gain_i = I;
	_gain_d = D;
}

void RateControl::setSaturationStatus(const Vector3<bool> &saturation_positive,
				      const Vector3<bool> &saturation_negative)
{
	_control_allocator_saturation_positive = saturation_positive;
	_control_allocator_saturation_negative = saturation_negative;
}

void RateControl::setPositiveSaturationFlag(size_t axis, bool is_saturated)
{
	if (axis < 3) {
		_control_allocator_saturation_positive(axis) = is_saturated;
	}
}

void RateControl::setNegativeSaturationFlag(size_t axis, bool is_saturated)
{
	if (axis < 3) {
		_control_allocator_saturation_negative(axis) = is_saturated;
	}
}

Vector3f RateControl::update(const Vector3f &rate, const Vector3f &rate_sp, const Vector3f &angular_accel,
			     const float dt, const bool landed, rate_ctrl_terms_s *terms)
{
	// angular rates error
	Vector3f rate_error = rate_sp - rate;
	_roll_rate_error = PX4_ISFINITE(rate_error(0)) ? rate_error(0) : 0.f;
	_roll_i_delta_raw = 0.f;
	_roll_i_delta_pre_imax = 0.f;
	_roll_i_delta_accepted = 0.f;
	_roll_i_update_enabled = false;

	// PID control with feed forward
	const Vector3f torque = _gain_p.emult(rate_error) + _rate_int - _gain_d.emult(angular_accel) + _gain_ff.emult(rate_sp);

	if (terms != nullptr) {
		_gain_p.emult(rate_error).copyTo(terms->p_term);
		_rate_int.copyTo(terms->i_term);
		(-_gain_d.emult(angular_accel)).copyTo(terms->d_term);
		_gain_ff.emult(rate_sp).copyTo(terms->ff_term);
		torque.copyTo(terms->output);
	}

	// update integral only if we are not landed
	if (!landed) {
		updateIntegral(rate_error, dt);
	}

	return torque;
}

void RateControl::updateIntegral(Vector3f &rate_error, const float dt)
{
	for (int i = 0; i < 3; i++) {
		if (i == 0) {
			float raw_i_factor = rate_error(i) / math::radians(400.f);
			raw_i_factor = math::max(0.0f, 1.f - raw_i_factor * raw_i_factor);
			const float raw_delta = raw_i_factor * _gain_i(i) * rate_error(i) * dt;

			if (PX4_ISFINITE(raw_delta)) {
				_roll_i_delta_raw = raw_delta;
				const float raw_accum_candidate = _roll_i_raw_drive_accum + raw_delta;

				if (PX4_ISFINITE(raw_accum_candidate)) {
					_roll_i_raw_drive_accum = raw_accum_candidate;
				}
			}
		}

		// prevent further positive control saturation
		if (_control_allocator_saturation_positive(i)) {
			rate_error(i) = math::min(rate_error(i), 0.f);
		}

		// prevent further negative control saturation
		if (_control_allocator_saturation_negative(i)) {
			rate_error(i) = math::max(rate_error(i), 0.f);
		}

		// I term factor: reduce the I gain with increasing rate error.
		// This counteracts a non-linear effect where the integral builds up quickly upon a large setpoint
		// change (noticeable in a bounce-back effect after a flip).
		// The formula leads to a gradual decrease w/o steps, while only affecting the cases where it should:
		// with the parameter set to 400 degrees, up to 100 deg rate error, i_factor is almost 1 (having no effect),
		// and up to 200 deg error leads to <25% reduction of I.
		float i_factor = rate_error(i) / math::radians(400.f);
		i_factor = math::max(0.0f, 1.f - i_factor * i_factor);

		// Perform the integration using a first order method
		float rate_i = _rate_int(i) + i_factor * _gain_i(i) * rate_error(i) * dt;

		// do not propagate the result if out of range or invalid
		if (PX4_ISFINITE(rate_i)) {
			if (i == 0) {
				_roll_i_delta_pre_imax = rate_i - _rate_int(i);
				const float shadow_candidate = _roll_i_shadow_no_imax + _roll_i_delta_pre_imax;

				if (PX4_ISFINITE(shadow_candidate)) {
					_roll_i_shadow_no_imax = shadow_candidate;
				}

				_roll_i_update_enabled = true;
			}

			if (i == 0 && _roll_i_transfer_context_enabled) {
				const RollILimits limits = computeRollILimits(_lim_int(i), _roll_i_transfer_context_raw,
						_roll_i_headroom_release_ratio);

				if (limits.valid) {
					_rate_int(i) = math::constrain(rate_i, limits.lower, limits.upper);
				}

			} else {
				_rate_int(i) = math::constrain(rate_i, -_lim_int(i), _lim_int(i));
			}

			if (i == 0) {
				_roll_i_delta_accepted = _rate_int(i) - (rate_i - _roll_i_delta_pre_imax);
			}
		}
	}
}

void RateControl::resetRollIntegralDiagnostics()
{
	_roll_rate_error = 0.f;
	_roll_i_delta_raw = 0.f;
	_roll_i_delta_pre_imax = 0.f;
	_roll_i_delta_accepted = 0.f;
	_roll_i_shadow_no_imax = 0.f;
	_roll_i_raw_drive_accum = 0.f;
	_roll_i_update_enabled = false;
}

void RateControl::getRateControlStatus(rate_ctrl_status_s &rate_ctrl_status)
{
	rate_ctrl_status.rollspeed_integ = _rate_int(0);
	rate_ctrl_status.pitchspeed_integ = _rate_int(1);
	rate_ctrl_status.yawspeed_integ = _rate_int(2);
	rate_ctrl_status.rollspeed_error = _roll_rate_error;
	rate_ctrl_status.rollspeed_integ_delta_raw = _roll_i_delta_raw;
	rate_ctrl_status.rollspeed_integ_delta_pre_imax = _roll_i_delta_pre_imax;
	rate_ctrl_status.rollspeed_integ_delta_accepted = _roll_i_delta_accepted;
	rate_ctrl_status.rollspeed_integ_shadow_no_imax = _roll_i_shadow_no_imax;
	rate_ctrl_status.rollspeed_integ_raw_drive_accum = _roll_i_raw_drive_accum;
	rate_ctrl_status.rollspeed_integ_update_enabled = _roll_i_update_enabled;
}
