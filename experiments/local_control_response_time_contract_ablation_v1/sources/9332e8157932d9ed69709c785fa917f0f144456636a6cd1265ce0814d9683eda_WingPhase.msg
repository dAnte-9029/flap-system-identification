uint64 timestamp                # time since system start (microseconds)

float32 phase_rad               # Hall-indexed flapping phase wrapped to [0, 2*pi) [rad]
float32 phase_unwrapped_rad     # Hall-indexed phase since the most recent Hall zero [rad]
float32 phase_deg               # Hall-indexed flapping phase wrapped to [0, 360) [deg]
float32 phase_unwrapped_deg     # Hall-indexed phase since the most recent Hall zero [deg]
float32 phase_sin               # sin(phase_rad)
float32 phase_cos               # cos(phase_rad)
float32 flap_frequency_hz       # flapping frequency [Hz]
uint32 encoder_position_raw     # AS5600 raw encoder position [0, 4095]
int64 encoder_total_count       # accumulated encoder counts from driver start
bool phase_valid                # true when a Hall zero has been latched and flap ratio is valid
