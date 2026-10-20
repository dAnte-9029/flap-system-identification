/****************************************************************************
 *
 *   Copyright (c) 2024 PX4 Development Team. All rights reserved.
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

#include "AS5600.hpp"
#include "AS5600Math.hpp"

#include <px4_platform_common/module.h>

using namespace time_literals;

static constexpr float kMinValidFlapFrequencyHz = 0.2f;

AS5600::AS5600(const I2CSPIDriverConfig &config) :
	I2C(config),
	I2CSPIDriver(config)
{
}

int AS5600::probe()
{
	uint16_t angle_raw{};

	if (read_raw_angle(angle_raw)) {
		return PX4_OK;
	}

	return PX4_ERROR;
}

bool AS5600::read_raw_angle(uint16_t &angle_raw)
{
	// RAW_ANGLE register: high byte at 0x0C, low byte at 0x0D
	uint8_t reg = 0x0C;
	uint8_t buf[2] {};

	const int ret = transfer(&reg, 1, buf, 2);

	if (ret != PX4_OK) {
		return false;
	}

	angle_raw = (static_cast<uint16_t>(buf[0]) << 8 | buf[1]) & 0x0FFF;
	return true;
}

bool AS5600::read_angle(float &angle_rad)
{
	uint16_t raw{};

	if (!read_raw_angle(raw)) {
		return false;
	}

	static constexpr float kScale = 2.f * M_PI_F / 4096.f;
	angle_rad = static_cast<float>(raw) * kScale;
	return true;
}

void AS5600::update_flap_ratio_param()
{
	if (_param_flap_ratio_handle == PARAM_INVALID) {
		return;
	}

	float ratio = _flap_ratio;

	if (param_get(_param_flap_ratio_handle, &ratio) == PX4_OK) {
		if (PX4_ISFINITE(ratio) && ratio > FLT_EPSILON) {
			_flap_ratio = ratio;
		}
	}
}

int AS5600::init()
{
	int ret = I2C::init();

	if (ret != PX4_OK) {
		return ret;
	}

	if (probe() != PX4_OK) {
		PX4_ERR("AS5600 not responding");
		return PX4_ERROR;
	}

	_param_flap_ratio_handle = param_find("FLAP_RATIO");

	update_flap_ratio_param();
	PX4_INFO("effective FLAP_RATIO: %.2f", static_cast<double>(_flap_ratio));

	ScheduleOnInterval(10_ms); // 100 Hz
	return PX4_OK;
}

void AS5600::RunImpl()
{
	uint16_t angle_raw{};

	if (!read_raw_angle(angle_raw)) {
		PX4_DEBUG("AS5600 read failed");
		return;
	}

	float angle_rad = static_cast<float>(angle_raw) * (2.f * M_PI_F / 4096.f);

	const hrt_abstime now = hrt_absolute_time();

	if (_param_flap_ratio_handle != PARAM_INVALID && (now - _last_param_update) > 1_s) {
		update_flap_ratio_param();
		_last_param_update = now;
	}

	// Initialize on first valid sample
	if (_last_read == 0) {
		_last_angle_rad = angle_rad;
		_last_read = now;
		_last_pos = angle_raw;
		_pos_initialized = true;
		return;
	}

	const float dt = (now - _last_read) * 1e-6f;

	// Reject unreasonable dt values
	if (dt <= 0.f || dt > 0.5f) {
		_last_angle_rad = angle_rad;
		_last_read = now;
		return;
	}

	float dtheta = angle_rad - _last_angle_rad;

	// position delta (0..4095) -> total counts
	int32_t dpos = static_cast<int32_t>(angle_raw) - static_cast<int32_t>(_last_pos);

	if (_pos_initialized) {
		if (dpos > 2048) {
			dpos -= 4096;

		} else if (dpos < -2048) {
			dpos += 4096;
		}

		_total_count += dpos;
	}

	// unwrap across 2*pi: constrain delta to (-pi, pi)
	if (dtheta > M_PI_F) {
		dtheta -= 2.f * M_PI_F;

	} else if (dtheta < -M_PI_F) {
		dtheta += 2.f * M_PI_F;
	}

	const float omega = dtheta / dt; // rad/s
	const float rpm_raw = omega * (60.f / (2.f * M_PI_F));

	// simple low-pass filter on rpm estimate
	const float alpha = 0.2f;

	if (!PX4_ISFINITE(_rpm_estimate)) {
		_rpm_estimate = rpm_raw;

	} else {
		_rpm_estimate = _rpm_estimate + alpha * (rpm_raw - _rpm_estimate);
	}

	// publish RPM
	rpm_s rpm_msg{};
	rpm_msg.timestamp = now;
	rpm_msg.rpm_raw = rpm_raw;
	rpm_msg.rpm_estimate = _rpm_estimate;
	_rpm_pub.publish(rpm_msg);

	flap_frequency_s flap_frequency{};
	flap_frequency.timestamp = now;
	float flap_frequency_hz = NAN;

	if (PX4_ISFINITE(_rpm_estimate) && PX4_ISFINITE(_flap_ratio) && (_flap_ratio > FLT_EPSILON)) {
		// Flap-frequency consumers only care about magnitude, not rotation direction.
		flap_frequency_hz = as5600::flap_frequency_from_rpm(_rpm_estimate, _flap_ratio);
		flap_frequency_hz = (flap_frequency_hz >= kMinValidFlapFrequencyHz) ? flap_frequency_hz : 0.f;
	}

	flap_frequency.frequency_hz = flap_frequency_hz;
	_flap_frequency_pub.publish(flap_frequency);

	_last_angle_rad = angle_rad;
	_last_read = now;
	_last_pos = angle_raw;
	_pos_initialized = true;

	// publish encoder count
	encoder_count_s enc{};
	enc.timestamp = now;
	enc.device_id = get_device_id();
	enc.total_count = _total_count;
	enc.position_raw = angle_raw;
	_encoder_pub.publish(enc);

	// debug_vect: x = angle, y = rpm_estimate, z = rpm_raw
	debug_vect_s dbg{};
	dbg.timestamp = now;
	strncpy(dbg.name, "AS5600ANG", sizeof(dbg.name));
	dbg.name[sizeof(dbg.name) - 1] = '\0';
	dbg.x = angle_rad;
	dbg.y = _rpm_estimate;
	dbg.z = rpm_raw;

	_debug_pub.publish(dbg);
}

void AS5600::print_status()
{
	I2CSPIDriverBase::print_status();
	PX4_INFO("last angle: %.3f rad", static_cast<double>(_last_angle_rad));
	PX4_INFO("rpm est=%.1f flap_hz=%.2f", static_cast<double>(_rpm_estimate),
		 static_cast<double>(as5600::flap_frequency_from_rpm(_rpm_estimate, _flap_ratio)));
}
