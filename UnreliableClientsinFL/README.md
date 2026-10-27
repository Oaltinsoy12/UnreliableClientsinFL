# UnreliableClientsinFL
This code is related to the paper titled: Federated Learning with Unreliable Clients: Performance Analysis and Mechanism Design
Due to the limited size, the related datasets are not uploaded, which can be found in the paper.

## Uplink SINR / SNR

Computed with `sinr.py`:
```
python sinr.py 001_8 003_8 005_8 001_16 003_16 005_16 --noise-std 0.01,0.03,0.05,0.01,0.03,0.05 --format markdown
```
Over an over-the-air (analog transmission) uplink, unit channel gain. Noise is applied
independently to each user's weights *before* saving, so M summed users contribute
`M * noise_std^2` of aggregate noise power (incoherent sum) — that noise energy is also
subtracted back out of the raw observed power before it's counted as signal:

- **SINR** — from `first_w_locals*.pt` (pre-defense, all users including malicious):
  `P_signal_true(reliable) / (P_interference(malicious) + M*noise_std^2)`
- **SNR** — from `latest_w_locals*.pt` (post-defense, malicious already excluded):
  `P_signal_true(reliable) / (M*noise_std^2)` (no interference term)

`001`/`003`/`005` vary the noise std applied to the weights (0.01 / 0.03 / 0.05); `_8`/`_16`
is the total user count (2 or 1 of which are malicious, per variant).

| run | variant | malicious | noise_std | SINR_dB | SNR_dB |
|---|---|---|---|---|---|
| 001_8 | 2-malicious | 4,5 | 0.01 | 7.35 | 27.06 |
| 001_8 | 1-malicious | 6 | 0.01 | 11.82 | 27.43 |
| 003_8 | 2-malicious | 4,5 | 0.03 | 2.73 | 23.72 |
| 003_8 | 1-malicious | 6 | 0.03 | 7.01 | 23.85 |
| 005_8 | 2-malicious | 4,5 | 0.05 | 0.36 | 23.38 |
| 005_8 | 1-malicious | 6 | 0.05 | 4.33 | 23.37 |
| 001_16 | 2-malicious | 4,5 | 0.01 | 14.72 | 27.35 |
| 001_16 | 1-malicious | 6 | 0.01 | 18.01 | 27.46 |
| 003_16 | 2-malicious | 4,5 | 0.03 | 9.74 | 21.16 |
| 003_16 | 1-malicious | 6 | 0.03 | 12.38 | 21.12 |
| 005_16 | 2-malicious | 4,5 | 0.05 | 6.47 | 19.66 |
| 005_16 | 1-malicious | 6 | 0.05 | 8.92 | 19.68 |

![SINR vs SNR across runs](sinr_comparison.png)

### Closed-form scalar approximation

Letting `n` = total users, `w` = per-element RMS amplitude of a "typical" reliable user
(`w = sqrt(mean_reliable P_i)`), `w_malicious` = same for a typical malicious user, and
`N_malicious` the number of malicious users:

```
SINR_dB = 10*log10( [((n - N_malicious)*w)^2 - P_noise] / (N_malicious*w_malicious^2 + P_noise) )
SNR_dB  = 10*log10( [((n - N_malicious)*w)^2 - P_noise] /  P_noise )
```

Two details matter here, both because malicious/noise sources are **independent** across
users and so add incoherently (power sums, not amplitude):
- the malicious term is `N_malicious * w_malicious^2` (sum of powers), **not**
  `(N_malicious * w_malicious)^2` (square of the summed amplitude, which would assume the
  attackers combine coherently);
- `P_noise` is subtracted from the numerator too, since noise is baked into the raw
  reliable-signal magnitude before it's ever observed.

This is a scalar idealization — it assumes all `n - N_malicious` reliable vectors point in
exactly the same direction, so `‖Σw_k‖ = Σ‖w_k‖`. `sinr.py` computes the coherent sum from
the actual per-user vectors instead (which aren't perfectly aligned in practice), so its
output is slightly below this formula's value — e.g. for `001_8` 2-malicious, the closed
form gives 8.30 dB vs. the script's exact 7.35 dB.

Computed with `python sinr.py <run_folders> --noise-std <...> --closed-form --format markdown`:

| run | variant | malicious | noise_std | SINR_dB | SNR_dB |
|---|---|---|---|---|---|
| 001_8 | 2-malicious | 4,5 | 0.01 | 8.30 | 27.10 |
| 001_8 | 1-malicious | 6 | 0.01 | 12.31 | 27.47 |
| 003_8 | 2-malicious | 4,5 | 0.03 | 6.82 | 23.82 |
| 003_8 | 1-malicious | 6 | 0.03 | 9.60 | 23.96 |
| 005_8 | 2-malicious | 4,5 | 0.05 | 6.37 | 23.48 |
| 005_8 | 1-malicious | 6 | 0.05 | 8.68 | 23.49 |
| 001_16 | 2-malicious | 4,5 | 0.01 | 15.25 | 27.46 |
| 001_16 | 1-malicious | 6 | 0.01 | 18.37 | 27.57 |
| 003_16 | 2-malicious | 4,5 | 0.03 | 12.67 | 21.58 |
| 003_16 | 1-malicious | 6 | 0.03 | 14.52 | 21.57 |
| 005_16 | 2-malicious | 4,5 | 0.05 | 11.66 | 20.23 |
| 005_16 | 1-malicious | 6 | 0.05 | 13.01 | 20.29 |

![SINR vs SNR across runs — closed-form approximation](sinr_comparison_closedform.png)
