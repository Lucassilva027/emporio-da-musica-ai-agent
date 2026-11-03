# Empório da Música — Agente de Atendimento (AI Engineer Challenge)

Protótipo de **agente de atendimento por mensagens de texto** para a Empório da Música (loja fictícia de instrumentos em Campo Grande/MS), desenvolvido como entrega do Desafio Técnico de AI Engineer da Artefact.

O assistente **Tom** responde em português com tom informal-profissional, consulta o **catálogo/estoque/pedidos em SQLite** e o **Manual de Políticas (PDF) via RAG híbrido**, e recusa com educação perguntas fora do escopo da loja.

---

## Como rodar

### 1. Pré-requisitos

- Python 3.10+ (testado em 3.14)
- (Opcional) chave gratuita de LLM — **Groq** ou **Gemini**

### 2. Instalação

```bash
cd "Desafio Técnico AI Engineer 2026"
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

### 3. Configurar o LLM (opcional, mas recomendado)

O projeto funciona **sem nenhuma API key** graças ao motor local híbrido. Para Function Calling com LLM (melhor fluência), edite o `.env` — o ideal é cadastrar **Groq e Gemini juntos** (failover automático se a Groq estourar o free tier):

| Provedor | Modelo padrão | Onde obter a chave | Variável |
|----------|---------------|--------------------|----------|
| **Groq** (prioridade) | `qwen/qwen3.8-27b` | https://console.groq.com/keys | `GROQ_API_KEY` |
| **Google Gemini** (failover) | `gemini-2.5-flash` | https://aistudio.google.com/apikey | `GEMINI_API_KEY` |
| Local | — | Nenhuma | `LLM_PROVIDER=local` |

- `LLM_PROVIDER=auto` (padrão): usa a primeira key disponível na ordem Groq → Gemini; sem keys, cai no motor local.
- Só OpenAI-compatible **Groq** e **Gemini** entram no projeto (sem OpenAI paga / Ollama).
- Após alterar o `.env`, **reinicie** o Streamlit/CLI para recarregar as keys.
- Diagnóstico rápido: `python scripts/ping_llm.py`

### 4. Interfaces

**CLI (Rich):**

```bash
python cli.py
python cli.py --provider local --show-tools
python cli.py --provider groq
python cli.py --provider gemini
```

Comandos na CLI: `/help`, `/reset`, `/tools on|off`, `/sair`.

**Web (Streamlit, estilo chat):**

```bash
streamlit run app.py
```

Na sidebar: escolha do provedor (`auto` / `groq` / `gemini` / `local`) e status do failover cloud.

### 5. Testes

```bash
python -m pytest tests/ -q
```

### 6. Regenerar exemplos de conversa

```bash
python scripts/generate_examples.py
```

Os arquivos ficam em `examples/`.

---

## Arquitetura

```
Cliente (CLI / Streamlit)
        │
        ▼
┌───────────────────────┐
│   EmporioAgent (Tom)  │  ← System Prompt + histórico SQLite
│  Function Calling /   │
│  Motor Local ReAct    │
└──────────┬────────────┘
           │ tools
     ┌─────┴──────┐
     ▼            ▼
 StoreToolkit   PolicyRAGEngine
 (SQLite ETL)   (PDF híbrido)
     │            │
 products.csv   políticas_da_loja.pdf
 orders.csv     (chunk por seção +
 promotions…     TF-IDF + MiniLM)
```

### Abordagem do agente (híbrida)

| Camada | Papel |
|--------|-------|
| **Function Calling / ReAct** | Quando há LLM (Groq/Gemini), o modelo decide quais tools chamar e sintetiza a resposta, com failover entre provedores. |
| **Motor local determinístico** | Classifica intenção (produto, pedido, política, promo, frete, fora de escopo…) e orquestra as **mesmas tools** — útil para demo zero-config e fallback se a API cair. |
| **RAG híbrido** | Políticas: embeddings densos (`all-MiniLM-L6-v2`) + TF-IDF + keyword boosting por seção do manual. |
| **Tools SQL** | Preços, estoque, promoções, pedidos e elegibilidade de troca/garantia calculados deterministicamente (sem “alucinar” números). |

**Por que híbrido?**  
Políticas são texto semiestruturado → RAG. Catálogo/pedidos são tabulares e mudam por linha → tools/SQL. Misturar os dois evita inventar preço/estoque e ainda permite explicar regras do manual com base em trechos recuperados.

### Persona e prompt

- Persona **Tom**, alinhada à Seção 7.1 (informal, acolhedor, “amigo que entende de música”).
- Regras críticas injetadas no system prompt: escopo só de instrumentos, PIX 5% não cumulativo com promoção, faixas de parcela mínima, LGPD, prazos de 7/30/90 dias.
- Data de referência configurável (`REFERENCE_DATE=2026-03-23`) para avaliar prazos de forma estável com a base de pedidos.

### Persistência

Histórico e contexto do cliente (nome, último pedido) em SQLite (`storage/emporio_da_musica.db`), permitindo retomar a sessão na CLI (`--session`) e na UI.

---

## Tratamento de dados

Pipeline ETL em `src/database.py` (versão `2.0`):

1. Normalização de texto (sem acento) para busca tolerante.
2. `effective_status` derivado: `available` | `out_of_stock` | `discontinued` | `coming_soon`.
3. Auditoria em `data_quality_audit` para inconsistências conhecidas, por exemplo:
   - Produto `active` com estoque 0 (ex.: Giannini GF-3D).
   - Divergências nome×descrição/specs em IDs 135–144 (marcas/cordas/teclas conflitantes).
   - Diferença `total_brl` do pedido vs soma dos itens.
   - Categorias cadastradas sem produtos (sopro / cordas orquestrais).
4. Promoções ativas aplicadas ao preço efetivo; PIX 5% **só** quando não há promoção (Seção 6.2).
5. Elegibilidade de devolução/troca/garantia calculada na tool `check_order_status`.

**Suposição documentada:** a data “hoje” do agente é `REFERENCE_DATE` (padrão 2026-03-23), imediatamente após o último pedido da base, para demonstrar pedidos entregues antigos (fora do prazo de 7 dias) e pedidos recentes/em trânsito.

---

## Exemplos de interação

| Arquivo | Cenário |
|---------|---------|
| [`examples/01_catalogo_violoes_ate_1000.md`](examples/01_catalogo_violoes_ate_1000.md) | Violões disponíveis até R$ 1.000 |
| [`examples/02_endereco_loja.md`](examples/02_endereco_loja.md) | Endereço / dados da loja |
| [`examples/03_preco_takamine_gd20.md`](examples/03_preco_takamine_gd20.md) | Preço do Takamine GD20 |
| [`examples/04_devolucao_arrependimento_pedido.md`](examples/04_devolucao_arrependimento_pedido.md) | **Não trivial:** política de arrependimento + pedido #1 |
| [`examples/05_fora_escopo_acessorios.md`](examples/05_fora_escopo_acessorios.md) | Pedido de palhetas/cabos (fora de escopo) |

---

## Estrutura do repositório

```
├── app.py                 # UI Streamlit
├── cli.py                 # CLI Rich
├── requirements.txt
├── .env.example
├── data/                  # CSVs + políticas_da_loja.pdf
├── examples/              # 5 conversas de demonstração
├── scripts/
│   ├── generate_examples.py
│   └── ping_llm.py        # testa se a API key do .env responde
├── src/
│   ├── agent.py           # Orquestração LLM + motor local
│   ├── config.py
│   ├── database.py        # ETL + histórico
│   ├── prompts.py
│   ├── rag.py             # RAG híbrido do PDF
│   └── tools.py           # Function calling tools
└── tests/
```

---

## Segurança e LGPD (protótipo)

Medidas implementadas neste escopo:

- **Segredos:** `.env` e `storage/*.db` no `.gitignore`; apenas `.env.example` versionado.
- **Allowlist de tools:** nomes fora de `ALLOWED_TOOLS` são rejeitados (mitiga function-calling abusivo).
- **SQL parametrizado** em todas as consultas SQLite.
- **Minimização de PII:** e-mail/telefone mascarados nas respostas de pedidos; busca por telefone exige ≥ 8 dígitos.
- **Homônimos (LGPD):** nomes ambíguos (ex.: “Bruno”) não listam pedidos — pedem e-mail/telefone/nº do pedido.
- **Limites de input:** mensagem do usuário truncada (2000 chars); queries/top_k/valores monetários com teto.
- **Erros:** falhas de API/tools não vazam stack trace nem corpo HTTP ao cliente final.
- **Isolamento de sessão:** cada conversa tem `session_id` próprio no SQLite.

Isto **não** substitui autenticação forte (OTP/WhatsApp) em produção.

---

## Limitações conhecidas (e o que faria com mais tempo)

1. **Motor local** é bom para precisão factual, mas menos “conversacional” que um LLM grande; com chave Groq/Gemini a fluência sobe bastante.
2. **Identificação de cliente** por nome ainda é simples; em produção usaria OTP/WhatsApp Business + consentimento LGPD.
3. **Categorias vazias** (sopro/orquestrais) e inconsistências cadastrais 135–144 são reportadas, mas não há UI de curadoria para o time da loja.
4. **Frete interestadual** informa modalidades/prazos, sem cotação CEP real (não há API de Correios no escopo).
5. **Avaliação automática** cobre regras de negócio e segurança básica; adicionaria golden answers + tool-selection accuracy.
6. Integração real com WhatsApp Cloud API / fila de handoff para humano em reclamações (Seção 7.3 — retorno em 24h).
7. Consulta só por `order_id` ainda é um fator de conhecimento fraco — em produção exigiria 2FA.

---

## Uso de assistentes de código (workflow)

Utilizei o **Cursor** (agente Auto / Composer) como copiloto de implementação:

- Leitura e limpeza do PDF de políticas + exploração dos CSVs.
- Scaffolding da arquitetura híbrida (ETL → tools → RAG → agent → CLI/Streamlit).
- Escrita de testes e geração dos exemplos de conversa.
- Decisões de produto (persona Tom, `REFERENCE_DATE`, motor local zero-config, Groq como default gratuito) foram definidas e revisadas por mim; o assistente acelerou boilerplate e o refino de prompts.

**Revisão humana do histórico Git:** cada commit progressivo foi revisado por mim antes (e após) o push — conferindo mensagem, diff, escopo da mudança e se o estado do código fazia sentido para a etapa (dados → RAG → tools → agente → UI → testes/segurança → docs). Não aceitei commits “caixa-preta”: quando algo saía do combinado (ex.: bugs de parsing de preço, máscara LGPD, cobertura de testes), eu pedi correção e só então consolidei no repositório público.

O histórico Git, portanto, reflete iterações reais com supervisão minha em cada etapa, e não um dump único gerado automaticamente.

---

## Licença / aviso

Projeto acadêmico/seletivo com dados fictícios da Empório da Música. Não afiliado à Artefact além do contexto do processo seletivo.
