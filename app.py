#!/usr/bin/env python3
"""Interface Web estilo WhatsApp — Agente Empório da Música (Streamlit)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import EmporioAgent
from src.config import STORE_INFO, get_llm_config

st.set_page_config(
    page_title="Empório da Música — Atendimento",
    page_icon="🎸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Estilo leve (WhatsApp-inspired, sem cards genéricos excessivos) ----
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(165deg, #0f2a1f 0%, #1a3d2e 40%, #243b2f 100%);
      }
      [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 0.4rem 0.6rem;
      }
      h1, h2, h3, p, label, .stMarkdown {
        color: #e8f5e9 !important;
      }
      .brand {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.8rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.2rem;
      }
      .tagline {
        opacity: 0.85;
        margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_agent() -> EmporioAgent:
    if "agent" not in st.session_state:
        provider = st.session_state.get("provider_choice", "auto")
        st.session_state.agent = EmporioAgent(
            session_id=st.session_state.get("session_id"),
            provider=None if provider == "auto" else provider,
        )
        st.session_state.session_id = st.session_state.agent.session_id
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"Olá! Eu sou o **Tom**, da {STORE_INFO['nome_fantasia']}. "
                    f"{STORE_INFO['slogan']}\n\n"
                    "Posso ajudar com catálogo, preços, estoque, promoções, "
                    "pedidos, trocas/devoluções e horários. Como posso te ajudar?"
                ),
            }
        ]
    return st.session_state.agent


with st.sidebar:
    st.markdown(f"### {STORE_INFO['nome_fantasia']}")
    st.caption(STORE_INFO["slogan"])
    st.markdown("---")
    provider_choice = st.selectbox(
        "Provedor LLM",
        options=["auto", "local", "groq", "gemini", "openai", "ollama"],
        index=0,
        help="auto usa a primeira API key disponível; senão cai no motor local.",
    )
    show_tools = st.toggle("Mostrar tool calls", value=True)

    if st.button("Nova conversa", use_container_width=True):
        if "agent" in st.session_state:
            st.session_state.agent.reset()
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.pop("agent", None)
        st.session_state.pop("messages", None)
        st.session_state.provider_choice = provider_choice
        st.rerun()

    st.session_state.provider_choice = provider_choice
    cfg = get_llm_config(None if provider_choice == "auto" else provider_choice)
    st.markdown("---")
    st.markdown(f"**Ativo:** `{cfg['provider']}`")
    st.caption(cfg["model"])
    st.markdown("---")
    st.markdown(
        f"**Endereço**\n\n{STORE_INFO['endereco']}\n\n"
        f"**WhatsApp** {STORE_INFO['whatsapp_atendimento']}"
    )
    st.markdown(
        "Seg–Sex 09:00–18:00 · Sáb 09:00–13:00 · Dom/feriados fechado"
    )

st.markdown(f'<div class="brand">{STORE_INFO["nome_fantasia"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="tagline">{STORE_INFO["slogan"]} · Assistente Tom</div>', unsafe_allow_html=True)

agent = get_agent()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎸" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if show_tools and msg.get("tools"):
            with st.expander("Ferramentas usadas"):
                for t in msg["tools"]:
                    st.code(f"{t['name']}({t['arguments']})", language="python")

prompt = st.chat_input("Digite sua mensagem…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎸"):
        with st.spinner("Consultando catálogo e políticas…"):
            result = agent.chat(prompt)
        st.markdown(result.reply)
        if show_tools and result.tool_calls:
            with st.expander("Ferramentas usadas"):
                for t in result.tool_calls:
                    st.code(f"{t['name']}({t['arguments']})", language="python")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.reply,
            "tools": result.tool_calls,
        }
    )
