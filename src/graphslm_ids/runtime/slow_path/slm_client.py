from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        context_length: int,
        max_new_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_ctx": context_length,
                "num_predict": max_new_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        parsed = json.loads(body)
        if "response" not in parsed:
            raise RuntimeError("Ollama response missing 'response' field.")
        return str(parsed["response"]).strip()
