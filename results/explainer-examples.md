# Explainer examples (not on the site)

Written by the configured review model from a code-computed digest of each predecessor diff. Most important
architectural change first, then size; model type omitted; readable numbers (each verified against the exact diff
value at its shown rounding); fields in review state the source conflict.

## Kimi-K3.1-Pro vs Kimi-K3

The layer mix shifts from 69 KDA linear attention plus 24 MLA (full) to 69 KDA linear attention plus 24 Gated DSA on MLA, and the sparse-attention token budget is 2,048 after being not stated for Kimi-K3. Total parameters go up from 2.7799T to 2.7801T, active parameters go up from 104.2B to 104.4B, and the growing cache per token goes up from 27 KB to 33 KB. For the weights, 5 weight groups were added with 179M parameters, and none were removed.

## LFM3-1.7B vs LFM2.5-1.2B-Instruct

The layer mix changed from 10 Gated short conv + 6 Global attention in the predecessor to 13 Gated short conv + 6 Global attention + 5 Sliding-window attention in the successor. Total parameters went up from 1.17B to 1.73B, and the number of layers went up from 16 to 24. Attention heads went down from 32 to 16, key-value heads went down from 8 to 4, head size went up from 64 to 128, and feed-forward width went down from 8,192 to 7,680. The predecessor's context length is in review: its config says 128,000, its card says 32,768, while the successor uses 262,144. The vocabulary went up from 65,536 to 128,000. Sliding window is not stated for LFM2.5-1.2B-Instruct and is 256 for the successor. Convolution kernel size went up from 3 to 7. Activation function is not stated for LFM2.5-1.2B-Instruct and is SiLU for the successor. Feed-forward type is not stated for LFM2.5-1.2B-Instruct and is gated for the successor. Regarding the weights, 6 weight groups added 46.6M parameters, while 3 removed 25.2M parameters.

## diffusiongemma-26B-A4B-it vs gemma-4-26B-A4B

The successor uses block diffusion as its generation method, with a diffusion canvas length of 256. Its active parameters went up to 3.84B from 3.82B in the predecessor. Regarding weights, the weight namespace changed from language_model to decoder, so added and removed tensor-group totals are not comparable.
