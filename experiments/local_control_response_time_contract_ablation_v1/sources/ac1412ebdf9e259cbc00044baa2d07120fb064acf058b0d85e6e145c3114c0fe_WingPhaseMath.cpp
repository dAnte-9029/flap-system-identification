#include "WingPhaseMath.hpp"

#include <float.h>
#include <math.h>

namespace wing_phase
{

CountInterpolationResult interpolate_count_at_timestamp(const EncoderSample &previous, const EncoderSample &current,
		uint64_t target_timestamp)
{
	if (previous.timestamp == 0 || current.timestamp == 0 || current.timestamp <= previous.timestamp) {
		return {};
	}

	if (target_timestamp < previous.timestamp || target_timestamp > current.timestamp) {
		return {};
	}

	if (target_timestamp == previous.timestamp) {
		return CountInterpolationResult{true, previous.total_count};
	}

	if (target_timestamp == current.timestamp) {
		return CountInterpolationResult{true, current.total_count};
	}

	const double dt = static_cast<double>(current.timestamp - previous.timestamp);
	const double alpha = static_cast<double>(target_timestamp - previous.timestamp) / dt;

	return CountInterpolationResult{
		true,
		previous.total_count + alpha * (current.total_count - previous.total_count)
	};
}

Result compute_phase(double encoder_total_count, double zero_count, float counts_per_cycle, bool zero_locked)
{
	static constexpr float kTwoPi = 6.28318530717958647692f;

	if (!zero_locked || !isfinite(counts_per_cycle) || counts_per_cycle <= FLT_EPSILON) {
		return {};
	}

	const double delta_count = encoder_total_count - zero_count;
	const double phase_unwrapped_rad = delta_count * (static_cast<double>(kTwoPi) / static_cast<double>(counts_per_cycle));
	double phase_rad = fmod(phase_unwrapped_rad, static_cast<double>(kTwoPi));

	if (phase_rad < 0.0) {
		phase_rad += static_cast<double>(kTwoPi);
	}

	Result result{};
	result.valid = true;
	result.phase_rad = static_cast<float>(phase_rad);
	result.phase_unwrapped_rad = static_cast<float>(phase_unwrapped_rad);
	return result;
}

} // namespace wing_phase
