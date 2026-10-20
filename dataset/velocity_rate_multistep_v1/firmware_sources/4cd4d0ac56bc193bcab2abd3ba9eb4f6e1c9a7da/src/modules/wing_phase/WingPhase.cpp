/****************************************************************************
 *
 *   Copyright (c) 2026 PX4 Development Team. All rights reserved.
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
 * @file WingPhase.cpp
 *
 * Compute Hall-indexed wing phase from encoder counts and Hall index pulses.
 */

#include "WingPhaseMath.hpp"

#include <px4_platform_common/getopt.h>
#include <px4_platform_common/log.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>

#include <math.h>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionInterval.hpp>
#include <uORB/topics/encoder_count.h>
#include <uORB/topics/flap_frequency.h>
#include <uORB/topics/hall_event.h>
#include <uORB/topics/parameter_update.h>
#include <uORB/topics/wing_phase.h>

using namespace time_literals;

class WingPhase : public ModuleBase<WingPhase>, public ModuleParams, public px4::ScheduledWorkItem
{
public:
	WingPhase();
	~WingPhase() override = default;

	static int task_spawn(int argc, char *argv[]);
	static int custom_command(int argc, char *argv[]);
	static int print_usage(const char *reason = nullptr);

	bool init();

private:
	static constexpr float kCountsPerRevolution = 4096.f;

	void Run() override;
	void updateParams() override;
	void updateEncoderSample(const encoder_count_s &encoder);
	void tryResolvePendingHall();

	uORB::Subscription _encoder_sub{ORB_ID(encoder_count)};
	uORB::Subscription _flap_frequency_sub{ORB_ID(flap_frequency)};
	uORB::Subscription _hall_sub{ORB_ID(hall_event)};
	uORB::SubscriptionInterval _param_update_sub{ORB_ID(parameter_update), 1_s};
	uORB::Publication<wing_phase_s> _phase_pub{ORB_ID(wing_phase)};

	double _zero_count{0.0};
	int64_t _last_total_count{0};
	uint32_t _last_position_raw{0};
	uint32_t _hall_pulse_count{0};
	hrt_abstime _last_encoder_timestamp{0};
	hrt_abstime _pending_hall_timestamp{0};
	float _last_flap_frequency_hz{NAN};
	float _counts_per_cycle{kCountsPerRevolution * 8.0f};
	bool _encoder_valid{false};
	bool _hall_locked{false};
	bool _hall_pending{false};
	wing_phase::EncoderSample _previous_encoder_sample{};
	wing_phase::EncoderSample _current_encoder_sample{};

	DEFINE_PARAMETERS(
		(ParamFloat<px4::params::FLAP_RATIO>) _param_flap_ratio
	)
};

WingPhase::WingPhase() :
	ModuleParams(nullptr),
	ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::hp_default)
{
}

bool WingPhase::init()
{
	updateParams();
	ScheduleOnInterval(5_ms);
	return true;
}

void WingPhase::updateParams()
{
	ModuleParams::updateParams();
	_counts_per_cycle = kCountsPerRevolution * _param_flap_ratio.get();
}

void WingPhase::updateEncoderSample(const encoder_count_s &encoder)
{
	if (_current_encoder_sample.timestamp != 0) {
		_previous_encoder_sample = _current_encoder_sample;
	}

	_current_encoder_sample.timestamp = encoder.timestamp;
	_current_encoder_sample.total_count = static_cast<double>(encoder.total_count);

	_last_total_count = encoder.total_count;
	_last_position_raw = encoder.position_raw;
	_last_encoder_timestamp = encoder.timestamp;
	_encoder_valid = true;
}

void WingPhase::tryResolvePendingHall()
{
	if (!_hall_pending || !_encoder_valid || _current_encoder_sample.timestamp == 0) {
		return;
	}

	double hall_count = 0.0;

	if (_pending_hall_timestamp == _current_encoder_sample.timestamp) {
		hall_count = _current_encoder_sample.total_count;

	} else {
		if (_previous_encoder_sample.timestamp == 0) {
			return;
		}

		const auto interpolation = wing_phase::interpolate_count_at_timestamp(_previous_encoder_sample,
				   _current_encoder_sample, _pending_hall_timestamp);

		if (!interpolation.valid) {
			return;
		}

		hall_count = interpolation.total_count;
	}

	static constexpr float kHallRearmPhaseRad = M_PI_F / 4.f;
	const wing_phase::Result phase_before_reset = wing_phase::compute_phase(hall_count, _zero_count, _counts_per_cycle,
							   _hall_locked);

	if (!_hall_locked
	    || (phase_before_reset.valid && fabsf(phase_before_reset.phase_unwrapped_rad) > kHallRearmPhaseRad)) {
		_zero_count = hall_count;
		_hall_locked = true;
	}

	_hall_pending = false;
}

void WingPhase::Run()
{
	if (should_exit()) {
		exit_and_cleanup();
		return;
	}

	if (_param_update_sub.updated()) {
		parameter_update_s parameter_update{};
		_param_update_sub.copy(&parameter_update);
		updateParams();
	}

	encoder_count_s encoder{};
	bool encoder_updated = false;

	while (_encoder_sub.update(&encoder)) {
		updateEncoderSample(encoder);
		encoder_updated = true;
	}

	flap_frequency_s flap_frequency{};

	while (_flap_frequency_sub.update(&flap_frequency)) {
		_last_flap_frequency_hz = flap_frequency.frequency_hz;
	}

	hall_event_s hall{};

	while (_hall_sub.update(&hall)) {
		if (hall.pulse_count != _hall_pulse_count) {
			_pending_hall_timestamp = hall.timestamp;
			_hall_pending = true;
		}

		_hall_pulse_count = hall.pulse_count;
	}

	tryResolvePendingHall();

	if (!_encoder_valid || !encoder_updated) {
		return;
	}

	const wing_phase::Result phase = wing_phase::compute_phase(_last_total_count, _zero_count, _counts_per_cycle, _hall_locked);

	wing_phase_s message{};
	message.timestamp = _last_encoder_timestamp ? _last_encoder_timestamp : hrt_absolute_time();
	message.flap_frequency_hz = _last_flap_frequency_hz;
	message.encoder_position_raw = _last_position_raw;
	message.encoder_total_count = _last_total_count;
	message.phase_valid = phase.valid;

	if (phase.valid) {
		message.phase_unwrapped_rad = phase.phase_unwrapped_rad;
		message.phase_rad = phase.phase_rad;
		message.phase_deg = message.phase_rad * (180.f / M_PI_F);
		message.phase_unwrapped_deg = message.phase_unwrapped_rad * (180.f / M_PI_F);
		message.phase_sin = sinf(message.phase_rad);
		message.phase_cos = cosf(message.phase_rad);

	} else {
		message.phase_unwrapped_rad = NAN;
		message.phase_rad = NAN;
		message.phase_deg = NAN;
		message.phase_unwrapped_deg = NAN;
		message.phase_sin = NAN;
		message.phase_cos = NAN;
	}

	_phase_pub.publish(message);
}

int WingPhase::task_spawn(int argc, char *argv[])
{
	WingPhase *instance = new WingPhase();

	if (!instance) {
		PX4_ERR("alloc failed");
		return PX4_ERROR;
	}

	_object.store(instance);
	_task_id = task_id_is_work_queue;

	if (instance->init()) {
		return PX4_OK;
	}

	delete instance;
	_object.store(nullptr);
	_task_id = -1;
	return PX4_ERROR;
}

int WingPhase::custom_command(int argc, char *argv[])
{
	return print_usage("unknown command");
}

int WingPhase::print_usage(const char *reason)
{
	if (reason) {
		PX4_WARN("%s\n", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
Compute Hall-indexed flapping phase from encoder counts and Hall index pulses.

Hall pulses define the mechanical zero. AS5600 encoder counts provide the continuous
phase between Hall events. The published wing_phase topic is invalid until the first
Hall event has been observed.
)DESCR_STR");

	PRINT_MODULE_USAGE_NAME("wing_phase", "controller");
	PRINT_MODULE_USAGE_COMMAND("start");
	PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
	return 0;
}

extern "C" __EXPORT int wing_phase_main(int argc, char *argv[])
{
	return WingPhase::main(argc, argv);
}
