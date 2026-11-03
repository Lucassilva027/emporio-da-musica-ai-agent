#!/usr/bin/env python3
"""CLI interativa do Agente de Atendimento da Empório da Música."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Garante import do pacote src mesmo rodando de fora do projeto
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.agent import EmporioAgent
from src.config import STORE_INFO, get_llm_config

console = Console()


def print_banner(agent: EmporioAgent) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Loja", f"{STORE_INFO['nome_fantasia']} — {STORE_INFO['slogan']}")
    table.add_row("Assistente", "Tom (AI Agent)")
    table.add_row("Provedor", f"{agent.provider_name} / {agent.model_name}")
    table.add_row("Sessão", agent.session_id[:8] + "…")
    table.add_row(
        "Comandos",
        "/sair  /reset  /tools on|off  /help",
    )
    console.print(
        Panel(
            table,
            title="[bold]Empório da Música — Atendimento AI[/bold]",
            border_style="bright_blue",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI do Agente Empório da Música")
    parser.add_argument("--provider", default=None, help="auto|groq|gemini|local")
    parser.add_argument("--session", default=None, help="ID de sessão para retomar histórico")
    parser.add_argument("--show-tools", action="store_true", help="Exibe tool calls no terminal")
    args = parser.parse_args()

    show_tools = args.show_tools
    cfg = get_llm_config(args.provider)

    def on_tool(name: str, arguments: dict) -> None:
        if show_tools:
            console.print(f"[dim]🔧 tool → {name}({arguments})[/dim]")

    agent = EmporioAgent(
        session_id=args.session,
        provider=args.provider,
        on_tool_call=on_tool,
    )
    print_banner(agent)
    if cfg["provider"] == "local":
        console.print(
            "[yellow]Nenhuma API key detectada — usando motor local híbrido "
            "(embeddings + SQLite + RAG). Configure GROQ_API_KEY ou GEMINI_API_KEY no .env "
            "para Function Calling com LLM.[/yellow]\n"
        )

    while True:
        try:
            user_input = console.input("[bold green]Você>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nAté logo! Sua música começa aqui. 🎸")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in {"/sair", "/exit", "/quit", "sair"}:
            console.print("Até logo! Sua música começa aqui. 🎸")
            break
        if low in {"/reset", "/nova"}:
            agent.reset()
            console.print(f"[cyan]Nova sessão iniciada: {agent.session_id[:8]}…[/cyan]")
            continue
        if low.startswith("/tools"):
            parts = low.split()
            if len(parts) > 1 and parts[1] in {"on", "off"}:
                show_tools = parts[1] == "on"
                console.print(f"[cyan]Exibição de tools: {'ON' if show_tools else 'OFF'}[/cyan]")
            else:
                console.print("[cyan]Uso: /tools on|off[/cyan]")
            continue
        if low in {"/help", "/ajuda"}:
            console.print(
                Markdown(
                    "### Comandos\n"
                    "- `/sair` — encerra\n"
                    "- `/reset` — nova sessão (limpa histórico)\n"
                    "- `/tools on|off` — mostra/oculta tool calls\n\n"
                    "### Exemplos\n"
                    "- Quais violões até R$1000?\n"
                    "- Quanto custa o Takamine GD20?\n"
                    "- Qual o endereço da loja?\n"
                    "- Me arrependi da compra, posso devolver o pedido #1?\n"
                    "- Vocês vendem palhetas?"
                )
            )
            continue

        with console.status("[bold blue]Tom está consultando…[/bold blue]", spinner="dots"):
            result = agent.chat(user_input)

        console.print()
        console.print(
            Panel(
                Markdown(result.reply),
                title=f"[bold]Tom[/bold] [{result.provider}]",
                border_style="magenta",
            )
        )
        if show_tools and result.tool_calls:
            for tc in result.tool_calls:
                console.print(
                    f"[dim]✓ {tc['name']} → {tc.get('result_preview', '')[:120]}…[/dim]"
                )
        console.print()


if __name__ == "__main__":
    main()
