# Saved prototype results

Generated from the four imported records. This run does not refetch the Hub or tensor headers.

| Model | Parameters in saved header counts | Verified fields | Single-source fields | Conflicts |
|---|---:|---:|---:|---:|
| LiquidAI/LFM2.5-1.2B-Instruct | 1,170,340,608 | 5 | 8 | 1 |
| LiquidAI/LFM3-1.7B | 1,728,801,536 | 11 | 6 | 0 |
| moonshotai/Kimi-K3.1-Pro | 2,780,110,363,392 | 24 | 3 | 0 |
| moonshotai/Kimi-K3 | 2,779,931,837,184 | 20 | 3 | 0 |

Offline tests: 22 passed; 0 failures; 0 errors. Python 3.12.13.

Recorded Kimi DSA-indexer addition: 178,526,208 parameters.

The LFM2.5-1.2B-Instruct FFN conflict is excluded from its published field dictionary. The independent 30-model gold set is still open.
