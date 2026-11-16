# Hard rules

- Free tiers only. Gemini via GEMINI_API_KEY (.env locally, GitHub secrets in Actions). Model names and limits
  live in config/providers.yaml. Never hard-code model names, publishers or limits in Python.
- One Gemini budget: every call, experimental or not, counts against the daily limit in config/providers.yaml.
- Never let an LLM supply architecture facts. Facts come only from published, verified record fields. LLMs only
  arrange, draw or explain them.
- Never commit secrets, webhook URLs or absolute local paths.
- An external reference sheet may be used for comparison only, named at run time and never committed: never a
  source for published fields or gold labels. Never download or use third-party images.
- Only ArchWatch's own approved diagrams may be used as house-style references, never third-party images.
- One PR per task, CI green before merge, tests for every change.
- The daily (no LLM) and Mon/Wed/Fri diagram crons were approved by the owner on 2026-11-15. Generated
  diagrams still reach the gallery only through a reviewed PR.
- No Co-Authored-By trailers in commits.
