# Quadratic-2 conditional twist sweep v2

## Scope and contracts

This directed follow-up conditions the \(\kappa\)-\(\psi_\theta\) map on
\(A_{\rm tip}\in\{20^\circ,30^\circ,35^\circ,40^\circ\}\). It uses the active
canonical ratio-8 phase-fixed dataset, the active DeLaurier prior, the
correction-ready accepted-cycle keys, current EKF wind, 72 mechanical-phase
bins, and body-FRD forces. Only train and validation are loaded; test remains
sealed. Static twist offset is fixed at \(0^\circ\).

The grid contains \(4\times9\times15=540\) unique physical parameter hashes.
Exactly 187 hashes were reused from the v1 coarse/refined compact logs and 353
were newly evaluated through the same formal force pipeline. The result table
contains 2160 partition/component metric rows.

## Baseline gate

The legacy model reproduces the full-cycle \(F_x\) peak at \(177.5^\circ\) in
both train and validation. In the primary \(180^\circ\)–\(270^\circ\) search
interval its peak is \(182.5^\circ\). The phase-binned data peaks are
\(212.5^\circ\) for train and \(217.5^\circ\) for validation. Active-prior
parity is exact for both \(F_x\) and \(F_z\).

## Conditional-map result

At \(A_{\rm tip}=20^\circ\), the validation peak spans only
\(182.5^\circ\)–\(187.5^\circ\); no parameter pair reaches within \(10^\circ\)
of both train and validation data peaks.

At \(A_{\rm tip}=30^\circ\), the validation peak spans
\(182.5^\circ\)–\(212.5^\circ\). Two grid points reach a \(5^\circ\) discrete
peak error in both partitions, but both violate the configured validation
\(F_z\) RMSE gate. For example, \((\kappa,\psi_\theta)=(-1,-35^\circ)\)
has validation \(F_z\) RMSE \(8.686\) N, compared with the legacy
\(7.562\) N and the allowed \(8.318\) N.

At \(A_{\rm tip}=35^\circ\), 23 grid points reach a \(5^\circ\) discrete peak
error in both partitions, and five also pass the configured physical/waveform
gates. The most balanced lower-amplitude candidate is
\((A_{\rm tip},\kappa,\psi_\theta)=(35^\circ,-0.5,-25^\circ)\). Its train and
validation peaks are \(207.5^\circ\) and \(212.5^\circ\), its signed validation
first-harmonic error is \(7.01^\circ\), and its validation circular lag is
\(5^\circ\). Validation \(F_x/F_z\) RMSE is \(2.550/7.819\) N.

At \(A_{\rm tip}=40^\circ\), 33 points reach a \(5^\circ\) discrete peak error
in both partitions, and eight pass the physical/waveform gates. The leading
candidate remains \((40^\circ,0,-25^\circ)\), with train/validation peaks
\(207.5^\circ/212.5^\circ\), signed validation first-harmonic error
\(4.63^\circ\), validation circular lag \(5^\circ\), and validation
\(F_x/F_z\) RMSE \(2.523/7.513\) N.

Curvature is therefore not independently phase-setting, but it is consequential
once dynamic twist is nonzero. At fixed \(\psi_\theta\), changing \(\kappa\)
can span as much as \(5^\circ,25^\circ,30^\circ,\) and \(60^\circ\) of the
discrete validation peak phase at amplitudes \(20^\circ,30^\circ,35^\circ,\)
and \(40^\circ\), respectively. The condition maps show continuous diagonal
bands: \(\kappa\) and \(\psi_\theta\) jointly determine the phase while
\(\kappa\) also moves the \(F_z\) tradeoff.

## Is 40 degrees necessary?

No. The \(35^\circ\) map contains a stable, physically gated region that
matches the discrete train and validation peak phases within one 5-degree bin.
The best combined robust phase error is \(7.01^\circ\) at \(35^\circ\) and
\(5.00^\circ\) at \(40^\circ\). The \(2.01^\circ\) gain is below the
preregistered \(5^\circ\) threshold for an upper-bound diagnostic, so the
separate \(45^\circ/50^\circ\) sweep was not run.

The \(40^\circ\) candidate remains modestly better: relative to the selected
\(35^\circ\) candidate, validation \(F_x\) RMSE improves by \(0.028\) N,
\(F_z\) RMSE improves by \(0.306\) N, and the first-harmonic phase error
improves by \(2.38^\circ\). These differences do not establish that
\(40^\circ\) is physically required.

## Wind sensitivity

For both selected candidates, replacing current EKF wind with zero wind changes
the validation discrete peak error from \(5^\circ\) to \(20^\circ\). The
\(35^\circ\) candidate changes from \(F_x/F_z\) RMSE \(2.550/7.819\) N to
\(2.650/6.334\) N; the \(40^\circ\) candidate changes from
\(2.523/7.513\) N to \(2.630/6.018\) N. Thus the qualitative conclusion does
not fully reverse, but the peak-phase agreement is wind-sensitive and sits
exactly at the configured 15-degree reversal tolerance.

## Conclusion

The v1 conclusion remains **B**: spanwise distribution helps, but the dominant
phase correction comes from independent twist timing. This follow-up adds two
important qualifications. First, \(\kappa\) has clear value at nonzero
amplitude because it reshapes the reachable phase region and the \(F_x/F_z\)
tradeoff. Second, \(35^\circ\) already provides performance close to
\(40^\circ\), so the data do not support claiming that the upper amplitude
boundary is necessary. The zero-wind sensitivity prevents interpreting
\(\psi_\theta\) as a uniquely identified structural phase lag without further
wind/airflow validation.

## Reproduction

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python \
  scripts/run_quadratic2_conditional_sweep.py \
  --config configs/delaurier/quadratic2_conditional_sweep_v2.yaml \
  --stage baseline

/home/zn/anaconda3/envs/flap-train-gpu/bin/python \
  scripts/run_quadratic2_conditional_sweep.py \
  --config configs/delaurier/quadratic2_conditional_sweep_v2.yaml \
  --stage conditional --resume --workers 8

/home/zn/anaconda3/envs/flap-train-gpu/bin/python \
  scripts/run_quadratic2_conditional_sweep.py \
  --config configs/delaurier/quadratic2_conditional_sweep_v2.yaml \
  --stage report
```
