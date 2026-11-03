"""Configurações centrais do projeto Empório da Música AI Agent."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv

    # Caminho absoluto: funciona mesmo se o Streamlit/CLI iniciar noutro cwd
    load_dotenv(ROOT_DIR / ".env", override=False)
except ImportError:
    pass

DB_PATH = STORAGE_DIR / "emporio_da_musica.db"
POLICIES_PDF_PATH = DATA_DIR / "políticas_da_loja.pdf"

# Metadados oficiais da loja (extraídos do Manual de Políticas v2.1)
STORE_INFO = {
    "razao_social": "Empório da Música Instrumentos Musicais Ltda.",
    "nome_fantasia": "Empório da Música",
    "slogan": "Sua música começa aqui.",
    "cnpj": "12.345.678/0001-90",
    "endereco": "Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, CEP 79202-333",
    "telefone_principal": "(67) 3341-4444",
    "whatsapp_atendimento": "(67) 3321-4500",
    "email": "contato@emporiodamusica.com.br",
    "fundacao": 2008,
    "horario_seg_sex": "09:00 às 18:00",
    "horario_sabado": "09:00 às 13:00",
    "horario_domingo_feriados": "Fechado",
}


def get_reference_date() -> date:
    """Retorna a data de referência para cálculo de prazos de troca, devolução e garantia.

    Por padrão utiliza 2026-03-23 (logo após o último pedido registrado em orders.csv,
    em 2026-03-22), permitindo testar cenários reais de arrependimento (7 dias),
    troca por defeito (30 dias) e garantia legal (90 dias).
    """
    raw = os.getenv("REFERENCE_DATE", "2026-03-23").strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return date(2026, 3, 23)


def _provider_configs() -> dict[str, dict[str, str]]:
    """Monta o mapa de provedores com as chaves atuais do ambiente."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    return {
        "groq": {
            "provider": "groq",
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip(),
        },
        "gemini": {
            "provider": "gemini",
            "api_key": gemini_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        },
        "local": {
            "provider": "local",
            "api_key": "",
            "base_url": "",
            "model": "local-hybrid-react-v1 (SentenceTransformers + SQLite)",
        },
    }


def list_configured_cloud_providers() -> list[str]:
    """Provedores cloud com API key preenchida (ordem de preferência para failover)."""
    cfg = _provider_configs()
    ordered = ["groq", "gemini"]
    return [name for name in ordered if cfg[name].get("api_key")]


def get_failover_providers(primary: str | None = None) -> list[str]:
    """Cadeia de failover: primary (se cloud) + demais keys disponíveis, sem repetir."""
    configured = list_configured_cloud_providers()
    if not configured:
        return []
    if primary and primary in configured:
        return [primary] + [p for p in configured if p != primary]
    return configured


def get_llm_config(override_provider: str | None = None) -> dict[str, str]:
    """Resolve dinamicamente a configuração do provedor de LLM.

    Suporta:
    - groq (Gratuito: qwen/qwen3.8-27b)
    - gemini (Gratuito via Google AI Studio: gemini-2.5-flash)
    - local (Motor Híbrido Local Zero-Config com Embeddings + SQLite/RAG)
    """
    provider = (override_provider or os.getenv("LLM_PROVIDER", "auto")).strip().lower()
    configs = _provider_configs()

    if provider == "auto":
        configured = list_configured_cloud_providers()
        provider = configured[0] if configured else "local"

    return dict(configs.get(provider, configs["local"]))
