"""Small standard-request Gemini adapter with a local request ledger; no agent loop, no fallback."""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .artifacts import write_if_changed

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
IMAGE_TOKEN_ALLOWANCE = 4000


class ProviderError(RuntimeError):
    pass


def local_env(path):
    """Read ignored local secrets without overriding real environment variables."""
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Gemini:
    """One configured free model. Every call, by any tool, is reserved in one shared ledger before it is
    sent and counts against that model's configured daily limit (one budget, no per-run allowances)."""

    def __init__(
        self, config, ledger, *, role="diagram", section="diagram_gen", clock=time.time, wait=time.sleep, opener=None
    ):
        self.config, self.ledger, self.section = config, Path(ledger), config[section]
        self.role = role
        self.clock, self.wait = clock, wait
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open
        model = config["models"][role]
        if not config.get("free_only") or not model.get("free"):
            raise ProviderError("Only an explicitly configured free model is permitted")
        if config["endpoint"] != "https://generativelanguage.googleapis.com/v1beta":
            raise ProviderError("Unrecognized provider endpoint")
        self.key = os.environ.get(config["key_env"])
        if not self.key:
            raise ProviderError("Missing configured Gemini API key")

    def _request(self, suffix, payload):
        request = urllib.request.Request(
            self.config["endpoint"] + suffix,
            data=json.dumps(payload).encode(),
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
        )
        try:
            with self.opener(request, timeout=self.config["request_timeout_seconds"]) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise ProviderError("Provider response exceeds the size bound")
                return json.loads(data)
        except urllib.error.HTTPError as error:
            # Never expose request headers, key-bearing URLs or provider bodies.
            raise ProviderError(f"Gemini request failed (HTTP {error.code}); no fallback attempted") from None
        except (OSError, ValueError):
            raise ProviderError("Gemini request failed; credentials and transport details omitted") from None

    def _reserve(self, prompt_bytes):
        now = self.clock()
        entries = json.loads(self.ledger.read_text()) if self.ledger.exists() else []
        model = self.config["models"][self.role]
        rpm = min(self.section["requests_per_minute"], model["rpm"])
        # Bytes over-count tokens, so this is a conservative per-minute token bound.
        if prompt_bytes + self.section["max_output_tokens"] > model["tpm"]:
            raise ProviderError("Conservative per-minute token bound exceeded")
        same = [e for e in entries if e["model"] == model["id"]]
        if same:
            delay = max(0, 60 / rpm - (now - same[-1]["time"]))
            if delay:
                self.wait(delay)
                now = self.clock()
        day = datetime.fromtimestamp(now, ZoneInfo("America/Los_Angeles")).date().isoformat()
        if sum(e["day"] == day and e["model"] == model["id"] for e in entries) >= model["rpd"]:
            raise ProviderError("Configured free daily request cap reached")
        # Reserve before transport; a timeout may already have consumed a call.
        entries.append(dict(time=now, day=day, model=model["id"], experiment=self.section["experiment"]))
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(self.ledger, json.dumps(entries, indent=2) + "\n")
        return now

    def generate(self, system, prompt, images=()):
        """`images`: PNG bytes sent after the text (e.g. a render of the previous attempt)."""
        model = self.config["models"][self.role]
        # An image costs a bounded number of tokens; count it generously against the per-minute bound.
        started = self._reserve(len(system.encode()) + len(prompt.encode()) + IMAGE_TOKEN_ALLOWANCE * len(images))
        parts = [dict(text=prompt)] + [
            dict(inline_data=dict(mime_type="image/png", data=base64.b64encode(png).decode())) for png in images
        ]
        response = self._request(
            "/models/" + model["id"] + ":generateContent",
            dict(
                systemInstruction=dict(parts=[dict(text=system)]),
                contents=[dict(role="user", parts=parts)],
                generationConfig=dict(
                    temperature=self.config["temperature"], maxOutputTokens=self.section["max_output_tokens"]
                ),
            ),
        )
        candidates = response.get("candidates") or []
        if not candidates:
            raise ProviderError("Provider returned no candidate")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ProviderError("Provider returned no output text")
        return text, {
            "model": model["id"],
            "finish_reason": candidates[0].get("finishReason"),
            "provider_model_version": response.get("modelVersion"),
            "usage": response.get("usageMetadata", {}),
            "elapsed_seconds": round(self.clock() - started, 3),
        }
