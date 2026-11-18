# Commit hashes before and after the 16 November 2026 history rewrite

On 16 November 2026 the repository history was rewritten (git-filter-repo) to remove third-party reference data.
Every commit hash changed. Frozen evidence files keep the hashes they were written with, unchanged; this table
translates them. Living documents were updated to the new hashes.

| Before | After | Commit |
|---|---|---|
| `5977d090797cad34f68645ae6eea47d3b9186716` | `785cac02c20af15e14fbf8307b42fd1cf90a1eac` | newest-backfill baseline code (merge of PR #8) |
| `a0a3f88d07d2bfd2274b976118b0a4629f9a48db` | `9caeb747d517e7a72a3d7a3866cdf6ac11ff54ee` | newest-release held-out labels frozen |
| `46c15850a0057a9c953770e01c526331fb7ba190` | `e33f8c747aa454da451510f00aebbc6691b367cf` | newest-release first-run records and results |
| `af7d814970231a472438d4159b2d33a5788fc582` | `840b22d20bce3ad3ac0502ff11e3905eec29dd3c` | separate late-access Meta baseline frozen before fixes |
| `fad28a00f557e431c33ba96e963cfb80a118130c` | `0fa673f99d6c42ddee8adc860f39714532b1aa05` | ten development gold labels frozen |
| `63767a7732b09c0636fb86b3d0bb603cd70edc3c` | `5bbc1d08bedf40e83aef8e2583ba56daf2f8e1b4` | cited in `results/github-workflow-acceptance.json` |
| `a10f153a705bdcdadf355f1ceba0de828e31c263` | `dc5bd784e418d22b1898b141fff4b7a31df45b65` | cited in `results/github-workflow-acceptance.json` |
| `c9a14c35c6613645bc5fa120093b2d9fe8b60a19` | `621d756e1156ff3e8d705a6995663298c36fc88e` | cited in `results/github-workflow-acceptance.json` |
| `f93be2c1a6fb1b333b335358d36daa606a095d18` | `c4eb4c5a95268f5c648af035f541a5e50088770e` | cited in `results/github-workflow-acceptance.json` |
| `6a95cfb7bfb86737e7a865f7bcdbec15a86fabe8` | `c7784442b89f381c9f2fd2f4b10825f1f0e17723` | cited in `results/live-validation.json` |
| `a2320733c10635fe3bd5fb4846a4dbdc7f42fce0` | `05563cb0317d1da985eec7cf1dc486500b1b8ed7` | cited in `results/no-op-acceptance.json` |

Frozen files that still cite the old hashes: `results/newest-first-run/summary.json`, `results/newest-first-run/cohort.yaml`,
`evals/heldout/README.md` and the three acceptance JSON files above.
