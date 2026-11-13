# Saved prototype results

Generated from four saved records refreshed on 13 November 2026. This summary command does not refetch sources; the separate live run is recorded in [live-validation.json](live-validation.json).

| Model | Parameters in saved header counts | Verified fields | Single-source fields | Conflicts |
|---|---:|---:|---:|---:|
| LiquidAI/LFM2.5-1.2B-Instruct | 1,170,340,608 | 6 | 15 | 0 |
| LiquidAI/LFM3-1.7B | 1,728,801,536 | 11 | 14 | 0 |
| moonshotai/Kimi-K3.1-Pro | 2,780,110,363,392 | 24 | 19 | 0 |
| moonshotai/Kimi-K3 | 2,779,931,837,184 | 20 | 19 | 0 |

Offline tests: 38 passed; 0 failures; 0 errors. Python 3.12.13.

Recorded Kimi DSA-indexer addition: 178,526,208 parameters.

The LFM2.5-1.2B-Instruct FFN width now applies the model's auto-adjust convention and agrees with the weights. The independent 30-model gold set is still open.
