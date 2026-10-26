#pragma once

#include <cmath>
#include <float.h>

namespace as5600
{

inline float flap_frequency_from_rpm(float rpm, float flap_ratio)
{
	return (std::isfinite(rpm) && std::isfinite(flap_ratio) && (flap_ratio > FLT_EPSILON))
	       ? std::fabs(rpm) / (60.f * flap_ratio) : NAN;
}

} // namespace as5600
