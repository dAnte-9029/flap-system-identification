# Models

Own reusable model structures, input/output contracts, features, and inference. Research may replace architectures, add explicit slow/periodic state representations, alter prediction targets, or restructure modules as the approved design requires. No separate authorization stage is needed simply because the change is substantial.

Do not use future observations, target values, or labels in deployable input features. Inference should not silently inspect test split paths or fit normalization. Keep physical priors, calibrated parameters, and learned corrections identifiable when a model uses them; purely learned models do not require fictitious physics components.

Keep training/selection code in appropriate training workflows. Do not require a baseline switch, extra ablation, or backward-compatibility scaffold unless needed for the requested comparison or an actual downstream dependency.
