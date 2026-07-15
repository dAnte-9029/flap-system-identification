# Physics Guidelines

Every physics input and output must state frame, unit, sign, and moment reference. Keep raw baselines separate from calibration and learned correction, and never change physical parameters silently.

Physics behavior changes require a separately authorized stage and numerical regression evidence. Do not add training, hyperparameter search, or test-based adjustment here.
