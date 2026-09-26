# Frozen loss definition
For k=25 candidate (50 baseline), average over states 1..k, excluding t0.
Position: mean ||dp||²; velocity: mean ||dv/2||²; attitude: mean 4(1-dot(qhat,q)²)/0.35² with unit quaternions; rate: mean ||dw/2||²; phase: 0.1 mean(2-2cos(dphase)); frequency: 0.1 mean(df/3)². Continuation adds 0.2 mean(df)² in Hz² on the SAME 1..k interval.
Both stages retain frozen weighted lag2 velocity/rate increment errors from t0 to t2, with original scales/weights. Zero actuator regularization. No teacher forcing, intermediate detach, or loss reweighting.
Validation L25 and L50 both include the final-stage frequency term and lag2 terms. They are window-weighted auxiliary objectives; physical endpoint metrics retain equal-flight aggregation. Different original training losses are not directly comparable.
