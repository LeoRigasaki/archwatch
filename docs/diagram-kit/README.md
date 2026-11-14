# Gemini diagram test kit

Tests how well Gemini can draw LLM architecture diagrams from a verified spec.
Two routes, same spec, so you can compare them:

- **Route A, image:** an image model draws the PNG directly. This tests raw power.
- **Route B, SVG:** a text model writes SVG code. Editable, and checkable by code. This is the route ArchWatch would use in production.

## Files

| File | Use |
| --- | --- |
| `specs/kimi-k3.1-pro.yaml`, `specs/lfm3-1.7b.yaml` | The spec to attach. Every value comes from the model's own files, verified by ArchWatch |
| `specs/*.txt` | The same specs as `.txt`, in case AI Studio won't accept a `.yaml` upload |
| `prompts/01_system_instructions.txt` | Paste into **System instructions** (both routes) |
| `prompts/02_image_prompt.txt` | The message for Route A |
| `prompts/03_svg_prompt.txt` | The message for Route B |
| `prompts/04_review_prompt.txt` | Grade any result in a fresh chat |

## Route A: image (start here)

1. Open [aistudio.google.com](https://aistudio.google.com) and start a new chat.
2. Pick the newest **image** model in the model list (Nano Banana Pro, `gemini-3-pro-image-preview`, or whatever newer image model is listed).
3. Paste `01_system_instructions.txt` into **System instructions**.
4. In the run settings, if offered: aspect ratio **4:3**, resolution **2K** (try 4K after), output **image only**.
5. Attach `specs/lfm3-1.7b.yaml` (start with LFM3, which is simpler) and send `02_image_prompt.txt`.
6. Run it 3 times and keep all 3. Image models vary a lot from run to run.
7. Repeat with `specs/kimi-k3.1-pro.yaml`.

Follow-up edits to try in the same chat, one at a time: "Fix only the spelling of the labels", "Move the zoom panels so no leader line crosses a box". Check whether the rest of the image stays the same.

## Route B: SVG

1. New chat, pick the strongest Gemini **text** model (Pro, with thinking on).
2. Same system instructions, attach the same YAML, send `03_svg_prompt.txt`.
3. Save the reply as `lfm3.svg` and open it in a browser. If it has problems, send back a screenshot and ask it to fix exactly those problems.

## Grading

Run `04_review_prompt.txt` in a **new** chat with a text model, attaching the image and the YAML. Then check it yourself against the answer key below. The model's self-review can miss things.

### Answer key: LFM3-1.7B

- [ ] Title "LFM3 (1.7B)"
- [ ] Groups from bottom to top: **1 x** [ShortConv, ShortConv], **5 x** [Global GQA, ShortConv, Sliding-window GQA, ShortConv], **1 x** [Global GQA, ShortConv], adding up to **24**
- [ ] Zoom panels: ShortConv (2048 -> 6144, kernel 7, B·x then conv then ·C), GQA (16 Q / 4 KV, head 128, QK-norm before rotary, head-wise gate 2048 -> 16), SwiGLU FFN (2048 -> 7680 -> 2048)
- [ ] Global layers: partial rotary, theta 1,000,000. Sliding layers: window 256, theta 10,000
- [ ] Callouts: 256K context, 128K vocabulary, 2048 embedding, 12 KB cache per token
- [ ] Sandwich norms (post-mixer and post-FFN RMSNorm) shown
- [ ] No benchmark numbers, no invented parts

### Answer key: Kimi K3.1 Pro

- [ ] Title "Kimi K3.1 Pro (2.8T)"
- [ ] Groups: **1 x** [KDA + dense FFN, KDA, KDA, Gated DSA], **22 x** [KDA, KDA, KDA, Gated DSA], **1 x** [Gated DSA], adding up to **93**
- [ ] 69 KDA + 24 Gated DSA. Layer 1 dense FFN, layers 2-93 LatentMoE
- [ ] KDA panel: q/k/v 7168 -> 12288, Conv1d kernel 4 on q, k and v, beta 7168 -> 96, gated delta rule, output RMSNorm, output gate
- [ ] DSA panel: q path 7168 -> 1536 -> 96 x 192, KV 7168 -> 576 (512 latent + 64), indexer 32 x 128, top 2048, **NoPE (the word RoPE must not appear)**
- [ ] LatentMoE: 896 experts, top 16, 2 shared, 7168 -> 3584 -> 3072 -> 3584 -> 7168
- [ ] AttnRes mix nodes on the residual paths, with no invented internals
- [ ] Callouts: 1M context, 33 KB cache per token vs 5.7 MB for plain MHA

### Scoring

- **Accuracy:** count the wrong or invented labels and numbers. 0 is publishable; 1-3 can be fixed with edits; more than 3 fails.
- **Readability:** can you read every label at 100% zoom? Is there no overlap?
- Write down which route and settings won. If neither gets accuracy to 0, the answer for ArchWatch is: Gemini writes the spec, and code draws the picture.
