# Labels and Targets

Keep target semantics explicit: frame, unit, sign, timing, moment reference, and whether a label is a whole-aircraft effective wrench, kinematic rate, or another observable. Do not misdescribe whole-aircraft effective-wrench labels as measured wing-only loads.

Align labels, priors, and predictions by stable sample/log/time keys; equal row counts do not demonstrate alignment. New target definitions and reconstruction methods are valid research changes. Version the changed meaning and update consumers together; do not silently reinterpret an old target.

Use train data for fitted target-processing choices. Keep sealed test information out of design and selection.
