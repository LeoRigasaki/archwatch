# Explainer examples (not on the site)

Written by the configured review model from a code-computed digest of each record's predecessor diff only. Every
number is checked against the digest; direction words must agree with each change in their sentence; no claim
words, raw field keys or tensor paths. Thousands separators are added by code after generation.

## moonshotai/Kimi-K3.1-Pro vs moonshotai/Kimi-K3

The model type changed from kimi_k3 to kimi_k31. Layer mix shifted from 69 KDA linear attention plus 24 MLA (full) to 69 KDA linear attention plus 24 Gated DSA on MLA, and the DSA top-k value is 2,048, having been not published for moonshotai/Kimi-K3. Total parameters increased from 2,779,931,837,184 to 2,780,110,363,392. Active parameters increased from 104,189,614,848 to 104,368,141,056. The growing cache per token, excluding sliding-window and constant states, increased in bytes from 27,648 to 33,792. Regarding weights, 5 tensor groups were added, totaling 178,526,208 parameters, with none removed.

Checks passed; 1 call(s) for this version; 88 words.

## LiquidAI/LFM3-1.7B vs LiquidAI/LFM2.5-1.2B-Instruct

Total parameters increased from 1,170,340,608 to 1,728,801,536, with the model type reported as lfm3. Layers went up from 16 to 24, changing the layer mix from 10 Gated short conv + 6 Global attention to 13 Gated short conv + 6 Global attention + 5 Sliding-window attention. The conv kernel went up from 3 to 7, the vocabulary increased from 65,536 to 128,000, and the head dim (q/k) went up from 64 to 128. Meanwhile, attention heads decreased from 32 to 16, KV heads decreased from 8 to 4, and the FFN hidden size decreased from 8,192 to 7,680. Context length is not published for LiquidAI/LFM2.5-1.2B-Instruct and is 262,144 for the successor, sliding window is not published for LiquidAI/LFM2.5-1.2B-Instruct and is 256 for the successor, activation is not published for LiquidAI/LFM2.5-1.2B-Instruct and is silu for the successor, and gated FFN is not published for LiquidAI/LFM2.5-1.2B-Instruct and is yes for the successor. Regarding the model weights, 6 tensor groups were added comprising 46,598,912 parameters, and 3 tensor groups were removed comprising 25,166,592 parameters.

Checks passed; 1 call(s) for this version; 172 words.

## google/diffusiongemma-26B-A4B-it vs google/gemma-4-26B-A4B

The model type changed from gemma4 to diffusion_gemma. Active parameters went up from 3,822,530,590 to 3,840,375,612. The generation type changed from not published for google/gemma-4-26B-A4B to block_diffusion, and the canvas length changed from not published for google/gemma-4-26B-A4B to 256. Regarding the weights, the weight namespace changed from language_model to decoder, so added and removed tensor-group totals are not comparable.

Checks passed; 1 call(s) for this version; 59 words.
