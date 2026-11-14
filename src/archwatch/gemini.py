"""Small standard-request adapter for the explicit route experiment; no agent loop."""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .artifacts import write_if_changed


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
    def __init__(self, config, ledger, *, clock=time.time, wait=time.sleep, opener=None):
        self.config, self.ledger = config, Path(ledger)
        self.clock, self.wait = clock, wait
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open
        model = config["models"]["diagram"]
        if not config.get("free_only") or not model.get("free"):
            raise ProviderError("Only the explicitly configured free diagram model is permitted")
        if config["endpoint"] != "https://generativelanguage.googleapis.com/v1beta":
            raise ProviderError("Unrecognized provider endpoint")
        self.key = os.environ.get(config["key_env"])
        if not self.key:
            raise ProviderError("Missing configured Gemini API key")

    def _request(self, suffix, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.config["endpoint"] + suffix,
            data=body,
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
        )
        try:
            with self.opener(request, timeout=self.config["request_timeout_seconds"]) as response:
                data = response.read(4 * 1024 * 1024 + 1)
                if len(data) > 4 * 1024 * 1024:
                    raise ProviderError("Provider response exceeds the experiment bound")
                return json.loads(data)
        except urllib.error.HTTPError as error:
            # Never expose request headers, key-bearing URLs or provider bodies.
            raise ProviderError(
                f"Gemini request failed (HTTP {error.code}); no fallback or upgrade attempted"
            ) from None
        except (OSError, ValueError):
            raise ProviderError("Gemini request failed; credentials and transport details omitted") from None

    def preflight(self):
        return self._request("/models/" + self.config["models"]["diagram"]["id"])

    def generate(self, prompt, *, json_output=False):
        now = self.clock()
        entries = json.loads(self.ledger.read_text()) if self.ledger.exists() else []
        model = self.config["models"]["diagram"]
        experiment = self.config["route_test"]
        rpm = min(experiment["requests_per_minute"], model["rpm"])
        if (len(prompt.encode()) + self.config["max_output_tokens"]) * rpm > model["tpm"]:
            raise ProviderError("Conservative per-minute token bound exceeded")
        if entries:
            delay = max(0, 60 / rpm - (now - entries[-1]["time"]))
            if delay:
                self.wait(delay)
                now = self.clock()
        date = datetime.fromtimestamp(now, ZoneInfo("America/Los_Angeles"))
        week = f"{date.isocalendar().year}-{date.isocalendar().week:02}"
        if sum(e["day"] == date.date().isoformat() and e["model"] == model["id"] for e in entries) >= model["rpd"]:
            raise ProviderError("Configured free daily request cap reached")
        if sum(e["week"] == week for e in entries) >= self.config["weekly_generation_request_cap"]:
            raise ProviderError("Configured weekly generation request cap reached")
        if sum(e.get("experiment") == "route-test-v1" for e in entries) >= experiment["max_generation_requests"]:
            raise ProviderError("Route experiment request cap reached; repairs are disabled")
        # Reserve before transport; a timeout may already have consumed a call.
        entries.append(
            dict(time=now, day=date.date().isoformat(), week=week, model=model["id"], experiment="route-test-v1")
        )
        write_if_changed(self.ledger, json.dumps(entries, indent=2) + "\n")
        generation = dict(temperature=self.config["temperature"], maxOutputTokens=self.config["max_output_tokens"])
        if json_output:
            generation["responseMimeType"] = "application/json"
        response = self._request(
            "/models/" + model["id"] + ":generateContent",
            dict(contents=[dict(role="user", parts=[dict(text=prompt)])], generationConfig=generation),
        )
        candidates = response.get("candidates") or []
        if not candidates:
            raise ProviderError("Provider returned no candidate; experiment stopped without repair")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ProviderError("Provider returned no output text")
        return text, {
            "model": model["id"],
            "finish_reason": candidates[0].get("finishReason"),
            "provider_model_version": response.get("modelVersion"),
            "usage": response.get("usageMetadata", {}),
            "elapsed_seconds": round(self.clock() - now, 3),
        }
