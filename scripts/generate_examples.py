"""Gera os arquivos de exemplo de conversa usando o motor local."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import EmporioAgent

SCENARIOS = [
    (
        "01_catalogo_violoes_ate_1000",
        "Quais opções de violões disponíveis custando até R$1000?",
        "Consulta de catálogo com filtro de categoria + preço máximo + disponibilidade.",
    ),
    (
        "02_endereco_loja",
        "Qual o endereço da loja?",
        "Informações gerais da loja via RAG de políticas / metadados oficiais.",
    ),
    (
        "03_preco_takamine_gd20",
        "Quanto custa o Takamine GD20?",
        "Consulta pontual de preço, PIX e parcelamento de um modelo específico.",
    ),
    (
        "04_devolucao_arrependimento_pedido",
        "Me arrependi da minha compra, posso devolver meu pedido #1?",
        "Cenário não trivial: cruza política de arrependimento (7 dias) com dados reais do pedido #1.",
    ),
    (
        "05_fora_escopo_acessorios",
        "Vocês vendem palhetas e cabos de guitarra?",
        "Pergunta fora do escopo comercial (acessórios) — recusa educada conforme Seções 1 e 7.1.",
    ),
]


def main() -> None:
    out = ROOT / "examples"
    out.mkdir(exist_ok=True)

    for fname, question, description in SCENARIOS:
        agent = EmporioAgent(provider="local")
        result = agent.chat(question)
        tools_lines = []
        for t in result.tool_calls:
            tools_lines.append(f"- `{t['name']}` — args: `{t['arguments']}`")
        tools_md = "\n".join(tools_lines) if tools_lines else "_Nenhuma tool acionada._"

        content = f"""# {description}

**Provedor:** `{result.provider}` / `{result.model}`

## Cliente

> {question}

## Tom (Assistente Empório da Música)

{result.reply}

## Ferramentas acionadas

{tools_md}
"""
        path = out / f"{fname}.md"
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.name} | tools={[t['name'] for t in result.tool_calls]}")


if __name__ == "__main__":
    main()
