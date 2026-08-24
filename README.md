# market-data-masking-detection

**Detecting affine scaling and directional inversion in anonymized tick data.**

When quantitative firms share proprietary data under NDA, they routinely apply
sanitizing transformations before handing it to external auditors. Two are
particularly common:

- **Affine price scaling** — `P' = αP + β` (hides asset identity, breaks tick-size checks)
- **Directional inversion** — `P' = −P + C` (reverses all returns, inflates fake alpha)

Standard audit pipelines and backtesting engines fail silently under both.
This repository demonstrates the failure and provides detection tools.

---

## The Silent Failure

A momentum strategy that earns a Sharpe of ~0.3 on real data reports ~4.5
on the same data after directional inversion. The backtester sees "alpha."
There is none.

```bash
pip install yfinance pandas numpy
python demo.py
```

---

## Detection Methods

### 1. GCD Tick Recovery (affine scaling)
All valid prices are integer multiples of the fundamental tick size.
The GCD of all non-zero absolute price movements recovers the tick size
**exactly**, regardless of the scaling constant α.

```python
from masking_detector import detect
profile = detect(df, time_col="timestamp", price_col="close", volume_col="volume")
print(profile.summary_lines())
```

### 2. Bid-Ask Impossibility (directional inversion — definitive)
Market mechanics require `Ask > Bid` at all times. After inversion,
`Ask' < Bid'` in 100% of rows. No estimation bias. No calibration required.

### 3. Engle-Ng Sign Bias + Lagged Leverage (supplementary)
When Level-2 data is unavailable, the sign bias test and lagged leverage
cross-correlation `L(τ) = Corr(r_t, r²_{t+τ})` provide heuristic evidence —
subject to the leverage effect puzzle at intraday frequency
(Aït-Sahalia, Fan and Li, 2011).

Run the full diagnostic on your own data:

```bash
# Edit BASE_DIR in the script first
python diagnose_leverage_effect.py
```

---

## Files

| File | Description |
|---|---|
| `demo.py` | Backtesting failure demonstration on real AAPL data |
| `masking_detector.py` | 7-protocol masking detector (affine, log-vol, info bars, z-score, epoch shift, jitter, column hashing) |
| `diagnose_leverage_effect.py` | Leverage signal diagnostic for your own store |
| `Jindal_2026_Auditing_Anonymized_Tick_Data.pdf` | Full paper with mathematical proofs and NSE empirical results |

---

## Research Paper

**Auditing Anonymized Tick Data: A Bias-Robust Framework for Detecting
Affine Scaling and Directional Inversion in Intraday Markets**

Aryan Jindal (2026) — Working Paper

> [Read on SSRN →](https://ssrn.com/abstract=XXXXXXX)
*(update link after SSRN upload)*

---

## Independent Data Audits

I run zero-access independent audits on proprietary NSE and international
intraday data — structural quality checks delivered as a PDF report,
without requiring strategy or model access.

**Contact:** aryan11jindal@gmail.com · LinkedIn www.linkedin.com/in/aryan-jindal-78a1a62b8

---

## License

MIT — detection code is free to use and adapt.
The audit methodology and commercial report renderer are not included here.