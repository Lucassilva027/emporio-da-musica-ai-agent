"""Configurações centrais do projeto Empório da Música AI Agent."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

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


def get_llm_config(override_provider: str | None = None) -> dict[str, str]:
    """Resolve dinamicamente a configuração do provedor de LLM.

    Suporta:
    - groq (Gratuito e recomendado: llama-3.3-70b-versatile)
    - gemini (Gratuito via Google AI Studio: gemini-2.5-flash)
    - openai (gpt-4o-mini / gpt-4o)
    - ollama (Modelos locais via http://localhost:11434/v1)
    - local (Motor Híbrido Local Zero-Config com Embeddings + ReAct Determinístico)
    """
    provider = (override_provider or os.getenv("LLM_PROVIDER", "auto")).strip().lower()

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()

    if provider == "auto":
        if groq_key:
            provider = "groq"
        elif gemini_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"
        else:
            provider = "local"

    if provider == "groq":
        return {
            "provider": "groq",
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        }
    if provider == "gemini":
        return {
            "provider": "gemini",
            "api_key": gemini_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        }
    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": openai_key,
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        }
    if provider == "ollama":
        return {
            "provider": "ollama",
            "api_key": "ollama",
            "base_url": ollama_url,
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip(),
        }

    return {
        "provider": "local",
        "api_key": "",
        "base_url": "",
        "model": "local-hybrid-react-v1 (SentenceTransformers + SQLite)",
    }
