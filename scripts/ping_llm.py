"""Smoke test do provedor LLM configurado no .env (não imprime a chave)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from src.config import get_llm_config


def main() -> None:
    cfg = get_llm_config()
    print("provider:", cfg["provider"])
    print("model:", cfg["model"])
    print("api_key_set:", bool(cfg.get("api_key")))
    if cfg["provider"] == "local":
        print("RESULT: ainda no motor local — verifique GROQ_API_KEY / GEMINI_API_KEY no .env")
        return

    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": "Responda apenas: OK"}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=45.0,
        )
    except Exception as exc:
        print("RESULT: falha de rede/conexão:", type(exc).__name__, exc)
        return

    print("http_status:", resp.status_code)
    if resp.status_code >= 400:
        print("error_body:", resp.text[:500])
        print("RESULT: chave ou modelo rejeitados pela API")
        return

    content = resp.json()["choices"][0]["message"].get("content", "")
    print("reply:", content)
    print("RESULT: OK — API LLM funcionando")


if __name__ == "__main__":
    main()
