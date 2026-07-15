# Models Guidelines

This directory owns feature definitions, model structures, inference contracts, and only the serialization data structures required by inference. It must not read train, validation, or test paths, run training loops, select candidates, plot, or report results.

Normalization fitting remains in training and may use train data only. Keep raw physics, calibrated physics, structured discrepancy, and neural residual parameters separate. A migration must not change model structure or defaults; new models require a separately authorized research stage.
