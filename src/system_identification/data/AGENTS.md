# Data Guidelines

This directory owns resampling, general preprocessing, and split data structures. It does not own effective-wrench label reconstruction, physics models, or model training.

Never process silently across log or segment boundaries. Never fit normalization, filters, or selection parameters on test data. Preserve sample identity, time order, and explicit exclusion reasons.
