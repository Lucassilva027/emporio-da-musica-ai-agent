"""Núcleo do Agente de Atendimento da Empório da Música.

Arquitetura híbrida:
1. Function Calling ReAct via provedores OpenAI-compatíveis (Groq / Gemini / OpenAI / Ollama).
2. Motor Local Determinístico (Zero-Config) que classifica intenção e orquestra as mesmas tools
   quando nenhuma API key estiver disponível — garantindo avaliação imediata do protótipo.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from src.config import (
    STORE_INFO,
    get_failover_providers,
    get_llm_config,
    get_reference_date,
)
from src.database import StoreDatabase, normalize_text
from src.prompts import build_system_prompt
from src.tools import StoreToolkit, format_brl

MAX_USER_MESSAGE_CHARS = 2000
MAX_TOOL_RESULT_CHARS = 3500
LLM_HISTORY_MESSAGES = 6
LLM_HTTP_RETRIES = 2  # tentativas por provedor em 429 (evita tempestade de calls)


@dataclass
class AgentTurnResult:
    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "local"
    model: str = "local-hybrid-react-v1"
    session_id: str = ""


class EmporioAgent:
    """Agente de atendimento com Function Calling + fallback local determinístico."""

    def __init__(
        self,
        session_id: str | None = None,
        provider: str | None = None,
        db: StoreDatabase | None = None,
        toolkit: StoreToolkit | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.db = db or StoreDatabase()
        self.toolkit = toolkit or StoreToolkit(db=self.db)
        self._preferred_provider = provider
        self.llm = get_llm_config(provider)
        self.on_tool_call = on_tool_call
        self.db.ensure_session(self.session_id)
        self._messages: list[dict[str, Any]] = []
        self.last_llm_http_calls = 0
        self._bootstrap_history()

    def _bootstrap_history(self) -> None:
        """Recarrega histórico persistido da sessão (se houver)."""
        stored = self.db.get_session_messages(self.session_id, limit=40)
        for msg in stored:
            if msg["role"] in ("user", "assistant", "system"):
                self._messages.append({"role": msg["role"], "content": msg["content"]})

    @property
    def provider_name(self) -> str:
        return self.llm["provider"]

    @property
    def model_name(self) -> str:
        return self.llm["model"]

    def reset(self) -> None:
        self.db.clear_session(self.session_id)
        self.session_id = str(uuid.uuid4())
        self.db.ensure_session(self.session_id)
        self._messages = []

    def chat(self, user_message: str) -> AgentTurnResult:
        user_message = (user_message or "").strip()
        if not user_message:
            return AgentTurnResult(
                reply="Pode me mandar sua dúvida? Estou aqui para ajudar com instrumentos, pedidos e políticas da loja. 🎸",
                provider=self.provider_name,
                model=self.model_name,
                session_id=self.session_id,
            )

        if len(user_message) > MAX_USER_MESSAGE_CHARS:
            user_message = user_message[:MAX_USER_MESSAGE_CHARS]

        self.db.save_message(self.session_id, "user", user_message)
        self._messages.append({"role": "user", "content": user_message})

        self.last_llm_http_calls = 0
        if self.provider_name == "local" or not self.llm.get("api_key"):
            result = self._run_local_engine(user_message)
        else:
            result = self._run_with_provider_failover(user_message)

        self.db.save_message(
            self.session_id,
            "assistant",
            result.reply,
            tool_calls=result.tool_calls or None,
        )
        self._messages.append({"role": "assistant", "content": result.reply})
        result.session_id = self.session_id
        result.provider = self.provider_name if result.provider != "local" else result.provider
        return result

    def _run_with_provider_failover(self, user_message: str) -> AgentTurnResult:
        """Tenta o provedor atual e, em 429/erro, os demais com API key no .env."""
        chain = get_failover_providers(self.provider_name)
        if not chain:
            chain = [self.provider_name]

        original = dict(self.llm)
        last_exc: Exception | None = None
        tried: list[str] = []

        for provider_name in chain:
            cfg = get_llm_config(provider_name)
            if not cfg.get("api_key"):
                continue
            self.llm = cfg
            tried.append(provider_name)
            try:
                result = self._run_llm_function_calling(user_message)
                # Mantém o provedor que funcionou nesta sessão (até Nova conversa / restart)
                return result
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                # Só faz failover automático em rate limit / indisponibilidade temporária
                if not ("429" in err or "rate_limit" in err.lower() or "503" in err or "502" in err):
                    break

        self.llm = original
        fallback = self._run_local_engine(user_message)
        err_txt = str(last_exc or "")
        if "429" in err_txt or "rate_limit" in err_txt.lower():
            note = (
                "_(Obs.: todos os provedores cloud configurados atingiram limite/cota "
                f"({', '.join(tried) or 'nenhum'}). Respondi com o motor local; "
                "aguarde ~20s ou adicione outra chave no `.env`.)_"
            )
        else:
            note = (
                "_(Obs.: houve instabilidade no provedor de linguagem"
                + (f" após tentar: {', '.join(tried)}" if tried else "")
                + "; usei o motor local de backup.)_"
            )
        fallback.reply = f"{fallback.reply}\n\n{note}"
        return fallback

    # ==========================================================================
    # Function Calling via LLM (Groq / Gemini / OpenAI / Ollama)
    # ==========================================================================
    def _run_llm_function_calling(self, user_message: str) -> AgentTurnResult:
        customer_ctx = self.db.get_customer_context(self.session_id)
        system_prompt = build_system_prompt(customer_ctx)
        tools = self.toolkit.get_openai_tool_schemas()

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # Histórico curto: free tier da Groq tem pouco ITPM
        for m in self._messages[-LLM_HISTORY_MESSAGES:]:
            messages.append({"role": m["role"], "content": m["content"]})

        tool_trace: list[dict[str, Any]] = []

        # No máximo 1 rodada de tools + 1 resposta final (free tier Groq estoura fácil com 3+ calls)
        response = self._chat_completions(messages, tools=tools)
        choice = response["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            reply = (choice.get("content") or "").strip()
            return AgentTurnResult(
                reply=reply or "Pronto! Posso te ajudar com mais alguma coisa?",
                tool_calls=tool_trace,
                provider=self.provider_name,
                model=self.model_name,
                session_id=self.session_id,
            )

        assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if choice.get("content"):
            assistant_msg["content"] = choice["content"]
        messages.append(assistant_msg)

        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}

            if self.on_tool_call:
                self.on_tool_call(name, args)

            result = self.toolkit.execute_tool(name, args)
            tool_trace.append({"name": name, "arguments": args, "result_preview": _preview(result)})
            self._maybe_update_customer_context(result)

            compact = _compact_tool_result(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"call_{idx}_{name}",
                    "content": json.dumps(compact, ensure_ascii=False, default=str),
                }
            )

        final = self._chat_completions(messages, tools=None)
        reply = (final["choices"][0]["message"].get("content") or "").strip()
        return AgentTurnResult(
            reply=reply or "Consegui consultar os dados. Posso esclarecer mais algum detalhe?",
            tool_calls=tool_trace,
            provider=self.provider_name,
            model=self.model_name,
            session_id=self.session_id,
        )

    def _chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.llm["model"],
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 900,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.llm['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{self.llm['base_url'].rstrip('/')}/chat/completions"

        last_error = ""
        with httpx.Client(timeout=60.0) as client:
            for attempt in range(LLM_HTTP_RETRIES):
                self.last_llm_http_calls += 1
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    last_error = resp.text[:400]
                    # Free tier: uma espera curta; se persistir, o failover troca de provedor
                    wait_s = 1.2 * (attempt + 1)
                    reset = resp.headers.get("x-ratelimit-reset-tokens", "")
                    try:
                        if reset.endswith("ms"):
                            wait_s = max(wait_s, float(reset[:-2]) / 1000.0 + 0.2)
                        elif reset.endswith("s"):
                            wait_s = max(wait_s, float(reset[:-1]) + 0.2)
                    except ValueError:
                        pass
                    time.sleep(min(wait_s, 4.0))
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
                return resp.json()

        raise RuntimeError(f"HTTP 429: {last_error}")

    def _maybe_update_customer_context(self, tool_result: dict[str, Any]) -> None:
        orders = tool_result.get("orders") or []
        if orders:
            cust = orders[0].get("customer") or {}
            self.db.update_customer_context(
                self.session_id,
                {
                    "customer_id": cust.get("customer_id"),
                    "name": cust.get("name"),
                    # Persiste apenas versões mascaradas — evita PII em claro no SQLite de sessão
                    "email_masked": cust.get("email_masked"),
                    "phone_masked": cust.get("phone_masked"),
                    "city": cust.get("city"),
                    "last_order_id": orders[0].get("order_id"),
                },
            )

    # ==========================================================================
    # Motor Local Determinístico (Zero-Config / Fallback)
    # ==========================================================================
    def _run_local_engine(self, user_message: str) -> AgentTurnResult:
        intent = self._classify_intent(user_message)
        tool_trace: list[dict[str, Any]] = []
        reply = ""

        def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            if self.on_tool_call:
                self.on_tool_call(name, args)
            result = self.toolkit.execute_tool(name, args)
            tool_trace.append({"name": name, "arguments": args, "result_preview": _preview(result)})
            self._maybe_update_customer_context(result)
            return result

        if intent == "greeting":
            reply = (
                f"Olá! Eu sou o Tom, assistente da {STORE_INFO['nome_fantasia']}. "
                f"{STORE_INFO['slogan']}\n\n"
                "Posso te ajudar com catálogo e preços, disponibilidade, promoções, "
                "status de pedidos, trocas/devoluções, frete e horários da loja. "
                "O que você precisa hoje?"
            )
        elif intent == "out_of_scope":
            reply = (
                "Haha, eu curto uma boa conversa, mas meu forte mesmo é instrumento musical! 🎸\n"
                "Sou o assistente da Empório da Música em Campo Grande/MS. "
                "Se quiser, posso te ajudar a escolher um violão, guitarra, baixo, bateria, teclado ou ukulele, "
                "consultar um pedido ou tirar dúvidas de troca/garantia. Como posso te ajudar?"
            )
        elif intent == "accessory":
            result = call("search_products", {"query": user_message})
            reply = self._format_accessory_reply(result)
        elif intent == "store_info":
            result = call("search_store_policies", {"query": user_message, "top_k": 2})
            reply = self._format_store_info_reply(user_message, result)
        elif intent == "policy":
            result = call("search_store_policies", {"query": user_message, "top_k": 3})
            # Se mencionar pedido/arrependimento, tenta cruzar com dados do pedido
            order_hint = self._extract_order_id(user_message)
            name_hint = self._extract_customer_name(user_message)
            order_data = None
            if order_hint or name_hint or any(
                k in normalize_text(user_message)
                for k in ["meu pedido", "minha compra", "devolver meu", "trocar meu"]
            ):
                args: dict[str, Any] = {}
                if order_hint:
                    args["order_id"] = order_hint
                if name_hint:
                    args["customer_name"] = name_hint
                if args:
                    order_data = call("check_order_status", args)
            reply = self._format_policy_reply(user_message, result, order_data)
        elif intent == "order":
            args = {}
            oid = self._extract_order_id(user_message)
            name = self._extract_customer_name(user_message)
            email = self._extract_email(user_message)
            phone = self._extract_phone(user_message)
            if oid:
                args["order_id"] = oid
            if name:
                args["customer_name"] = name
            if email:
                args["customer_email"] = email
            if phone:
                args["customer_phone"] = phone
            # Se não há identificador, pede ao cliente
            if not args:
                ctx = self.db.get_customer_context(self.session_id)
                if ctx.get("last_order_id"):
                    args["order_id"] = ctx["last_order_id"]
                elif ctx.get("name"):
                    args["customer_name"] = ctx["name"]
            result = call("check_order_status", args)
            reply = self._format_order_reply(result)
        elif intent == "brands":
            cat = ""
            for official, aliases in {
                "Violões": ["violao", "violoes"],
                "Ukuleles": ["ukulele"],
                "Guitarras": ["guitarra"],
                "Baixos": ["baixo", "contrabaixo"],
                "Baterias e Percussão": ["bateria"],
                "Teclados e Pianos": ["teclado", "piano", "sintetizador"],
            }.items():
                if any(a in normalize_text(user_message) for a in aliases):
                    cat = official
                    break
            result = call(
                "list_brands",
                {"category": cat, "only_available": True},
            )
            reply = self._format_brands_reply(result)
        elif intent == "promotions":
            camp = self._extract_campaign(user_message)
            result = call(
                "get_promotions",
                {"include_expired": True, "campaign_name": camp},
            )
            reply = self._format_promotions_reply(result, camp)
        elif intent == "payment_shipping":
            # Tenta extrair valor; se houver menção a produto, busca primeiro
            amount = self._extract_price(user_message)
            city = self._extract_city(user_message) or "Campo Grande"
            has_promo = False
            if amount is None:
                prod = call("search_products", {"query": user_message, "only_available": False})
                products = prod.get("products") or []
                if products:
                    amount = products[0]["effective_price_brl"]
                    has_promo = products[0].get("active_promotion") is not None
                    reply_prefix = self._format_products_reply(prod, concise=True) + "\n\n"
                else:
                    amount = 0.0
                    reply_prefix = ""
            else:
                reply_prefix = ""
            pay = call(
                "calculate_payment_and_shipping",
                {
                    "amount_brl": amount,
                    "city": city,
                    "has_active_promotion": has_promo,
                    "is_large_instrument": any(
                        w in normalize_text(user_message)
                        for w in ["bateria", "piano", "contrabaixo", "teclado"]
                    ),
                },
            )
            reply = reply_prefix + self._format_payment_shipping_reply(pay)
        else:
            # product search (default)
            args = self._build_product_search_args(user_message)
            result = call("search_products", args)
            reply = self._format_products_reply(result)

            # Se o cliente também perguntou sobre parcelamento/PIX no mesmo turno
            if any(k in normalize_text(user_message) for k in ["pix", "parcel", "frete", "parcelar"]):
                products = result.get("products") or []
                if products:
                    p0 = products[0]
                    pay = call(
                        "calculate_payment_and_shipping",
                        {
                            "amount_brl": p0["effective_price_brl"],
                            "city": "Campo Grande",
                            "has_active_promotion": p0.get("active_promotion") is not None,
                        },
                    )
                    reply += "\n\n" + self._format_payment_shipping_reply(pay)

        return AgentTurnResult(
            reply=reply.strip(),
            tool_calls=tool_trace,
            provider="local",
            model="local-hybrid-react-v1",
            session_id=self.session_id,
        )

    def _classify_intent(self, text: str) -> str:
        n = normalize_text(text)

        greetings = ["ola", "olá", "oi", "bom dia", "boa tarde", "boa noite", "eai", "e ai", "hey"]
        if any(n == g or n.startswith(g + " ") or n.startswith(g + ",") for g in greetings) and len(n.split()) <= 4:
            return "greeting"

        out_of_scope_kw = [
            "receita", "futebol", "politica", "eleicao", "bitcoin", "codigo python",
            "programacao", "filme", "serie", "namoro", "horoscopo", "clima amanha",
        ]
        if any(k in n for k in out_of_scope_kw):
            return "out_of_scope"

        accessory_kw = [
            "palheta", "palhetas", "cabo p10", "cabos", "pedal", "pedais", "amplificador",
            "amplificadores", "case rígido", "case rigido", "capa para", "encordoamento",
            "jogo de cordas", "cordas avulsas", "baqueta", "capotraste", "correia",
        ]
        instrument_kw = ["violao", "guitarra", "baixo", "ukulele", "bateria", "teclado", "piano"]
        if any(k in n for k in accessory_kw) and not any(k in n for k in instrument_kw):
            return "accessory"

        policy_kw = [
            "devolver", "devolucao", "trocar", "troca", "arrependi", "arrependimento",
            "garantia", "defeito", "reembolso", "estorno", "posso devolver", "posso trocar",
            "lgpd", "privacidade", "excluir meus dados",
        ]
        # Políticas têm prioridade sobre "pedido" quando o cliente fala de troca/devolução,
        # para cruzar RAG + elegibilidade do pedido no mesmo turno.
        if any(k in n for k in policy_kw):
            return "policy"

        order_kw = [
            "pedido", "rastreio", "rastreamento", "tracking", "meu pedido", "status do",
            "onde esta", "quando chega", "codigo de rastreio",
        ]
        if any(k in n for k in order_kw) or re.search(r"\bpedido\s*#?\s*\d+\b", n):
            return "order"

        store_kw = [
            "endereco", "onde fica", "localizacao", "horario", "funciona", "abre", "fecha",
            "telefone", "whatsapp", "cnpj", "email da loja", "contato",
        ]
        if any(k in n for k in store_kw):
            return "store_info"

        # "quais marcas", "que marcas tem", "marcas vocês possuem" — sem citar marca específica
        brand_question = any(
            k in n
            for k in [
                "quais marcas",
                "que marcas",
                "quais marca",
                "marcas voces",
                "marcas voce",
                "marcas tem",
                "marcas possuem",
                "marcas trabalham",
                "marcas disponiveis",
                "lista de marcas",
            ]
        )
        if brand_question:
            return "brands"

        promo_kw = [
            "promocao", "promocoes", "desconto", "oferta", "black friday", "queima de estoque",
            "aniversario da loja", "semana do musico", "liquidacao",
        ]
        if any(k in n for k in promo_kw):
            return "promotions"

        pay_ship_kw = ["parcelar", "parcelamento", "formas de pagamento", "frete", "envio", "quanto fica o frete"]
        # Se só fala de pagamento/frete sem citar produto específico, trata como payment_shipping
        if any(k in n for k in pay_ship_kw) and not any(k in n for k in instrument_kw + ["takamine", "yamaha", "fender", "taylor", "martin"]):
            return "payment_shipping"

        return "product"

    def _build_product_search_args(self, text: str) -> dict[str, Any]:
        n = normalize_text(text)
        args: dict[str, Any] = {"query": text, "only_available": False}

        max_price = self._extract_price(text)
        if max_price is not None and any(k in n for k in ["ate", "abaixo", "menos de", "no maximo", "maximo"]):
            args["max_price"] = max_price
        elif max_price is not None and any(k in n for k in ["acima", "mais de", "a partir"]):
            args["min_price"] = max_price

        if "nylon" in n:
            args["strings_type"] = "nylon"
        elif "aco" in n or "steel" in n:
            args["strings_type"] = "aco"

        if any(k in n for k in ["eletrico", "eletroacustico", "com captacao", "plugado"]):
            args["has_electronics"] = True
        elif any(k in n for k in ["acustico puro", "sem captacao", "nao eletrico"]):
            args["has_electronics"] = False

        if any(k in n for k in ["disponivel", "disponiveis", "em estoque", "pronta entrega"]):
            args["only_available"] = True

        # Categoria explícita
        for cat, aliases in {
            "Violões": ["violao", "violoes"],
            "Ukuleles": ["ukulele", "ukuleles"],
            "Guitarras": ["guitarra", "guitarras"],
            "Baixos": ["baixo", "baixos", "contrabaixo"],
            "Baterias e Percussão": ["bateria", "baterias"],
            "Teclados e Pianos": ["teclado", "teclados", "piano", "sintetizador"],
        }.items():
            if any(re.search(rf"\b{a}\b", n) for a in aliases):
                args["category"] = cat
                break

        return args

    # ---- Extractors ----
    @staticmethod
    def _extract_order_id(text: str) -> int | None:
        m = re.search(r"(?:pedido\s*#?\s*|n[uú]mero\s*|order\s*#?\s*|#\s*)(\d{1,4})", text, re.I)
        if m:
            return int(m.group(1))
        m2 = re.search(r"\b(\d{1,2})\b", text)
        # Só aceita número solto se a mensagem for claramente sobre pedido
        if m2 and "pedido" in normalize_text(text):
            return int(m2.group(1))
        return None

    @staticmethod
    def _extract_email(text: str) -> str | None:
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        return m.group(0) if m else None

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 10:
            return digits[-11:] if len(digits) >= 11 else digits[-10:]
        return None

    @staticmethod
    def _extract_customer_name(text: str) -> str | None:
        patterns = [
            r"(?:meu nome [eé]\s+|me chamo\s+|sou (?:o|a)\s+|cliente\s+)([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})",
            r"(?:pedido d[oa]\s+|da cliente\s+|do cliente\s+)([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+){0,3})",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _extract_price(text: str) -> float | None:
        """Extrai valor monetário em BRL.

        Aceita: R$1000, R$ 1.000, 1.000,00, 1000 reais.
        Importante: não usar ``\\d{1,3}`` isolado como primeira alternativa,
        senão ``1000`` vira ``100``.
        """
        patterns = [
            r"r\$\s*(\d{1,3}(?:\.\d{3})+,\d{2})",      # R$ 1.000,00
            r"r\$\s*(\d{1,3}(?:\.\d{3})+)",            # R$ 1.000
            r"r\$\s*(\d+[.,]\d{2})",                   # R$ 1000,00 / R$ 1000.00
            r"r\$\s*(\d+)",                            # R$1000 / R$ 1000
            r"(\d{1,3}(?:\.\d{3})+,\d{2})\s*reais?",   # 1.000,00 reais
            r"(\d{1,3}(?:\.\d{3})+)\s*reais?",         # 1.000 reais
            r"(\d+[.,]\d{2})\s*reais?",                # 1000,00 reais
            r"(\d+)\s*reais?",                         # 1000 reais
        ]
        raw = None
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                raw = m.group(1)
                break
        if raw is None:
            return None
        if "," in raw and "." in raw:
            raw = raw.replace(".", "").replace(",", ".")
        elif "," in raw:
            raw = raw.replace(",", ".")
        elif "." in raw and len(raw.split(".")[-1]) == 3:
            raw = raw.replace(".", "")
        try:
            val = float(raw)
            n = normalize_text(text)
            if val < 50 and "r$" not in text.lower() and "real" not in n and "reais" not in n:
                return None
            return val
        except ValueError:
            return None

    @staticmethod
    def _extract_city(text: str) -> str | None:
        cities = [
            "Campo Grande", "Dourados", "Três Lagoas", "Tres Lagoas",
            "Corumbá", "Corumba", "Ponta Porã", "Ponta Pora",
        ]
        n = normalize_text(text)
        for c in cities:
            if normalize_text(c) in n:
                return c
        return None

    @staticmethod
    def _extract_campaign(text: str) -> str:
        n = normalize_text(text)
        mapping = {
            "black friday": "Black Friday",
            "aniversario": "Aniversário da Loja",
            "semana do musico": "Semana do Músico",
            "queima": "Queima de Estoque",
            "liquidacao": "Liquidação",
            "desconto exclusivo": "Desconto Exclusivo",
            "oferta especial": "Oferta Especial",
            "pascoa": "Páscoa",
        }
        for k, v in mapping.items():
            if k in n:
                return v
        return ""

    # ---- Formatters ----
    def _format_products_reply(self, result: dict[str, Any], concise: bool = False) -> str:
        if result.get("status") == "out_of_scope_accessory":
            return self._format_accessory_reply(result)

        if result.get("status") == "empty_category":
            return (
                f"A categoria **{result['category']}** faz parte do nosso cadastro, "
                "mas no momento não há nenhum instrumento dessa linha disponível no sistema. "
                "Quer que eu te mostre outras categorias? Temos violões, guitarras, baixos, "
                "baterias, teclados e ukuleles."
            )

        products = result.get("products") or []
        unavailable = result.get("unavailable_or_discontinued_matches") or []
        alts = result.get("suggested_alternatives_for_unavailable") or []

        if not products and not unavailable:
            return (
                "Não encontrei nenhum instrumento com esses filtros no catálogo atual. "
                "Quer tentar por marca (Yamaha, Takamine, Fender…), categoria ou faixa de preço?"
            )

        lines: list[str] = []
        available = [p for p in products if p.get("is_available")]
        shown = available if available else products

        if available:
            lines.append(f"Encontrei **{result.get('total_available_matching', len(available))}** opção(ões) disponível(is):")
        else:
            lines.append("Encontrei estes itens (atualmente indisponíveis):")

        for p in shown[:8 if not concise else 3]:
            lines.append(self._format_product_card(p))

        # Produtos específicos buscados que estão esgotados/descontinuados
        for u in unavailable[:2]:
            if u not in shown:
                status_label = {
                    "out_of_stock": "temporariamente sem estoque",
                    "discontinued": "descontinuado",
                    "coming_soon": "em breve (pré-venda / lançamento)",
                }.get(u["effective_status"], u["effective_status"])
                lines.append(
                    f"\n⚠️ **{u['name']}** — {status_label}. "
                    f"Preço de tabela: {_brl(u['original_price_brl'])}."
                )
                if u.get("data_quality_notes"):
                    lines.append(f"_Obs. cadastral: {u['data_quality_notes']}_")

        if alts:
            lines.append("\n**Alternativas disponíveis parecidas:**")
            for a in alts[:3]:
                lines.append(self._format_product_card(a, short=True))

        if not concise:
            lines.append("\nQuer que eu detalhe algum modelo, calcule o PIX/parcelamento ou o frete para sua cidade?")
        return "\n".join(lines)

    def _format_brands_reply(self, result: dict[str, Any]) -> str:
        brands = result.get("brands") or []
        if not brands:
            scope = result.get("resolved_category") or "catálogo"
            return (
                f"No momento não encontrei marcas com produtos disponíveis em **{scope}**. "
                "Quer que eu liste as marcas de outra categoria (violões, guitarras, teclados…)?"
            )

        header = "**Marcas disponíveis no catálogo"
        if result.get("resolved_category"):
            header += f" ({result['resolved_category']})"
        header += f":** {result.get('brands_count', len(brands))} marca(s)\n"

        lines = [header]
        for b in brands:
            price_bit = ""
            if b.get("min_price_brl") is not None and b.get("max_price_brl") is not None:
                if abs(b["min_price_brl"] - b["max_price_brl"]) < 0.01:
                    price_bit = f" — a partir de {_brl(b['min_price_brl'])}"
                else:
                    price_bit = f" — de {_brl(b['min_price_brl'])} a {_brl(b['max_price_brl'])}"
            cats = ", ".join(b.get("categories") or [])
            lines.append(
                f"- **{b['brand']}**: {b['product_count']} produto(s)"
                f"{price_bit}"
                + (f" ({cats})" if cats else "")
            )

        lines.append(
            "\nSe quiser, diga a marca (ex.: Yamaha, Fender, Takamine) ou a categoria "
            "que eu te mostro os modelos disponíveis."
        )
        return "\n".join(lines)

    def _format_product_card(self, p: dict[str, Any], short: bool = False) -> str:
        promo = p.get("active_promotion")
        stock = f"{p['stock_quantity']} em estoque" if p["is_available"] else p["effective_status"]
        inst = (p.get("installment_plan") or {}).get("summary") or ""
        # Escapa $ no resumo de parcelas (Streamlit / KaTeX)
        inst_safe = inst.replace("$", r"\$")

        if promo:
            price_block = (
                f"Preço: ~~{_brl(p['original_price_brl'])}~~ → **{_brl(p['effective_price_brl'])}** "
                f"({promo['discount_percent']:.0f}% OFF — {promo['campaign']})"
            )
        else:
            price_block = (
                f"Preço: **{_brl(p['effective_price_brl'])}** · "
                f"PIX: **{_brl(p['pix_price_brl'])}**"
            )

        if short:
            return f"- **{p['name']}** — {price_block} ({stock})"

        desc = (p.get("description") or "")[:140]
        lines = [
            f"\n- **{p['name']}** ({p['category_name']})",
            f"  - {price_block}",
        ]
        if inst_safe:
            lines.append(f"  - Parcelamento: {inst_safe}")
        lines.append(f"  - Estoque: {stock}")
        if desc:
            lines.append(f"  - {desc}…")
        if p.get("data_quality_notes"):
            lines.append(f"  - _Obs. cadastral: {p['data_quality_notes']}_")
        return "\n".join(lines)

    def _format_accessory_reply(self, result: dict[str, Any]) -> str:
        terms = ", ".join(result.get("matched_accessory_terms") or ["acessórios"])
        return (
            f"Entendi o interesse em **{terms}**, mas a Empório da Música trabalha "
            "**exclusivamente com instrumentos musicais** — não vendemos cordas avulsas, palhetas, "
            "cabos, cases, pedais ou amplificadores (política interna, Seções 1 e 7.1).\n\n"
            "Sugiro procurar uma loja parceira especializada em acessórios. "
            "Por aqui, posso te ajudar a escolher o instrumento ideal — violão, guitarra, baixo, "
            "bateria, teclado ou ukulele. Quer uma recomendação?"
        )

    def _format_store_info_reply(self, query: str, result: dict[str, Any]) -> str:
        n = normalize_text(query)
        info = STORE_INFO
        parts: list[str] = []

        if any(k in n for k in ["endereco", "onde fica", "localizacao", "cep"]):
            parts.append(
                f"Nosso endereço é **{info['endereco']}**.\n"
                f"Telefone/WhatsApp da loja: **{info['telefone_principal']}** "
                f"(atendimento WhatsApp: **{info['whatsapp_atendimento']}**).\n"
                f"E-mail: **{info['email']}**."
            )
        if any(k in n for k in ["horario", "abre", "fecha", "funciona", "expediente"]):
            parts.append(
                "**Horário de funcionamento:**\n"
                f"• Segunda a Sexta: {info['horario_seg_sex']}\n"
                f"• Sábado: {info['horario_sabado']}\n"
                f"• Domingo e feriados: {info['horario_domingo_feriados']}\n"
                "Em datas especiais (Black Friday, Aniversário da Loja, Natal) o horário pode ser estendido."
            )
        if any(k in n for k in ["telefone", "whatsapp", "contato", "email", "cnpj"]):
            parts.append(
                f"**Contatos:**\n"
                f"• Telefone: {info['telefone_principal']}\n"
                f"• WhatsApp atendimento: {info['whatsapp_atendimento']}\n"
                f"• E-mail: {info['email']}\n"
                f"• CNPJ: {info['cnpj']}"
            )

        if not parts:
            sections = result.get("retrieved_sections") or []
            if sections:
                top = sections[0]
                parts.append(
                    f"Sobre **{top['section_title']}** (Manual Seção {top['section_number']}):\n"
                    f"{top['content'][:700]}…"
                )
            else:
                parts.append(
                    f"Somos a {info['nome_fantasia']}, em {info['endereco']}. "
                    f"Horário: seg–sex {info['horario_seg_sex']}, sáb {info['horario_sabado']}."
                )

        parts.append("\nPosso te ajudar com mais alguma coisa — catálogo, pedido ou políticas?")
        return "\n\n".join(parts)

    def _format_policy_reply(
        self,
        query: str,
        rag_result: dict[str, Any],
        order_data: dict[str, Any] | None,
    ) -> str:
        n = normalize_text(query)
        sections = rag_result.get("retrieved_sections") or []
        lines: list[str] = []

        if any(k in n for k in ["arrepend", "devolver", "devolucao"]):
            lines.append(
                "**Direito de Arrependimento (compras online — Seção 4.1):**\n"
                "• Até **7 dias corridos** após o recebimento, sem precisar justificar.\n"
                "• Produto na embalagem original, sem sinais de uso, com acessórios e manuais.\n"
                "• Frete de devolução por conta da loja.\n"
                "• Reembolso na mesma forma de pagamento em até **10 dias úteis**."
            )
        if any(k in n for k in ["troca", "trocar", "preferencia", "defeito"]):
            lines.append(
                "**Trocas (Seção 4.2 e 4.3):**\n"
                "• Por **defeito de fabricação**: até **30 dias corridos** após a compra (na loja). "
                "Depois disso, acionar garantia do fabricante (podemos intermediar).\n"
                "• Por **preferência** (cor/modelo/tamanho): até **7 dias**, produto perfeito na embalagem original.\n"
                "• Não elegíveis: personalizações/setup sob encomenda, liquidação com 'venda final', "
                "e boquilhas de sopro (higiene)."
            )
        if "garantia" in n:
            lines.append(
                "**Garantia (Seção 8):**\n"
                "• Garantia legal CDC: **90 dias** a partir do recebimento.\n"
                "• Garantia do fabricante: tipicamente de 6 meses a 2 anos (ver certificado do produto).\n"
                "• Não cobre: desgaste natural, mau uso, quedas, umidade, modificações por terceiros."
            )

        if not lines and sections:
            top = sections[0]
            lines.append(
                f"Conforme o Manual (**Seção {top['section_number']} — {top['section_title']}**):\n"
                f"{top['content'][:900]}"
            )

        if order_data and order_data.get("status") == "ok" and order_data.get("orders"):
            o = order_data["orders"][0]
            elig = o.get("return_and_warranty_eligibility") or {}
            lines.append(
                f"\n**Aplicando ao seu Pedido #{o['order_id']}** ({o['status_pt']}, "
                f"cliente {o['customer']['name']}):\n"
                f"{elig.get('explanation', '')}\n"
                f"• Arrependimento 7 dias: {'✅ elegível' if elig.get('regret_return_7_days') else '❌ fora do prazo'}\n"
                f"• Troca por defeito 30 dias (loja): {'✅ elegível' if elig.get('defect_store_exchange_30_days') else '❌ fora do prazo / acionar fabricante'}\n"
                f"• Garantia legal 90 dias: {'✅ elegível' if elig.get('legal_warranty_90_days') else '❌ fora do prazo'}"
            )
        elif order_data and order_data.get("status") in ("missing_identifier", "ambiguous_customer_lgpd"):
            lines.append(f"\n{order_data.get('message', '')}")
        elif any(k in n for k in ["meu pedido", "minha compra"]) and not order_data:
            lines.append(
                "\nPara eu verificar se o **seu** pedido ainda está no prazo, me passe o número do pedido, "
                "seu nome completo, e-mail ou telefone (LGPD — Seção 9)."
            )

        lines.append(f"\n_Data de referência do sistema: {get_reference_date().isoformat()}._")
        lines.append("Quer que eu consulte um pedido específico?")
        return "\n\n".join(lines)

    def _format_order_reply(self, result: dict[str, Any]) -> str:
        status = result.get("status")
        if status != "ok":
            return result.get("message") or "Não consegui localizar o pedido com os dados informados."

        lines: list[str] = []
        for o in result.get("orders") or []:
            items_txt = ", ".join(
                f"{i['quantity']}x {i['product_name']}" for i in (o.get("items") or [])
            )
            track = o.get("tracking_code") or "ainda não gerado"
            lines.append(
                f"**Pedido #{o['order_id']}** — {o['status_pt']}\n"
                f"• Cliente: {o['customer']['name']} ({o['customer']['city']})\n"
                f"• Data: {o['order_date']} | Previsão/entrega: {o.get('estimated_or_actual_delivery') or '—'}\n"
                f"• Total: {_brl(o['total_brl'])} | Pagamento: {o['payment_method']}\n"
                f"• Itens: {items_txt}\n"
                f"• Rastreio: `{track}`"
            )
            if o.get("notes"):
                lines.append(f"• Observação: {o['notes']}")
            elig = o.get("return_and_warranty_eligibility") or {}
            if elig.get("explanation"):
                lines.append(f"• Políticas: {elig['explanation']}")
            lines.append("")

        lines.append("Precisa de ajuda com troca, devolução ou rastreio?")
        return "\n".join(lines)

    def _format_promotions_reply(self, result: dict[str, Any], campaign: str) -> str:
        active = result.get("active_promotions") or []
        expired = result.get("expired_promotions") or []
        lines: list[str] = []

        if active:
            lines.append(f"**Promoções ativas agora ({len(active)}):**")
            for p in active:
                lines.append(
                    f"• **{p['product_name']}** — ~~{_brl(p['original_price_brl'])}~~ → "
                    f"**{_brl(p['promotional_price_brl'])}** "
                    f"({p['discount_percent']:.0f}% OFF — {p['campaign']}) "
                    f"| status: {p['effective_status']}"
                )
        else:
            lines.append("No momento não há promoções ativas no sistema para o filtro informado.")

        if campaign and expired:
            lines.append(
                f"\nSobre a campanha **{campaign}**: encontrei registros **expirados** "
                "(conforme Seção 7.3, informamos com transparência e apresentamos o preço vigente):"
            )
            for p in expired[:5]:
                lines.append(
                    f"• {p['product_name']}: era {p['discount_percent']:.0f}% OFF "
                    f"(agora preço de tabela {_brl(p['original_price_brl'])})"
                )
        elif expired and not active:
            lines.append("\nAlgumas campanhas recentes já encerradas (amostra):")
            for p in expired[:3]:
                lines.append(f"• {p['campaign']} em {p['product_name']} — encerrada")

        lines.append(
            "\nLembrete: promoções **não são cumulativas** com o desconto de 5% do PIX (Seção 6.2)."
        )
        return "\n".join(lines)

    def _format_payment_shipping_reply(self, result: dict[str, Any]) -> str:
        pay = result.get("payment_options") or {}
        ship = result.get("shipping_options") or {}
        pix = pay.get("pix") or {}
        credit = pay.get("credit_card") or {}

        def _esc(text: Any) -> str:
            return str(text or "").replace("$", r"\$")

        lines = [
            f"**Pagamento para {_brl(result.get('base_amount_brl', 0))}:**",
            f"- PIX: {_esc(pix.get('description'))}",
            f"- Débito: {_brl((pay.get('debit_card') or {}).get('amount_brl', 0))} à vista",
            f"- Boleto: {_brl((pay.get('boleto') or {}).get('amount_brl', 0))} (compensa em até 3 dias úteis)",
            f"- Crédito: {_esc(credit.get('summary'))}",
            f"- {_esc(pay.get('split_payment_rule'))}",
            "",
            "**Frete:**",
        ]

        if "shipping_cost_brl" in ship:
            free = (
                "🎁 Frete grátis!"
                if ship.get("is_free_shipping")
                else _brl(ship["shipping_cost_brl"])
            )
            lines.append(
                f"- {ship.get('region')}: {free} | Prazo: {ship.get('estimated_days')} | "
                f"{ship.get('delivery_method')}"
            )
        else:
            lines.append(f"• {ship.get('region')}")
            for c in ship.get("carriers") or []:
                lines.append(f"  – {c['modality']}: {c['estimated_days']}")
            if ship.get("large_instrument_warning"):
                lines.append(f"• ⚠️ {ship['large_instrument_warning']}")

        return "\n".join(lines)


def _brl(value: float | int | None) -> str:
    """Atalho markdown-safe: escapa `$` para o Streamlit não abrir modo LaTeX."""
    return format_brl(value, markdown_safe=True)


def _preview(data: Any, limit: int = 280) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Reduz payload das tools para caber no rate limit (ITPM) da Groq free tier."""
    if not isinstance(result, dict):
        return {"value": str(result)[:500]}

    if result.get("error"):
        return {"error": result.get("error"), "message": result.get("message")}

    compact: dict[str, Any] = {"status": result.get("status", "ok")}

    if "products" in result:
        products = []
        for p in (result.get("products") or [])[:6]:
            products.append(
                {
                    "id": p.get("product_id"),
                    "name": p.get("name"),
                    "category": p.get("category_name"),
                    "price": p.get("effective_price_brl"),
                    "pix": p.get("pix_price_brl"),
                    "installments": (p.get("installment_plan") or {}).get("summary"),
                    "stock": p.get("stock_quantity"),
                    "available": p.get("is_available"),
                    "status": p.get("effective_status"),
                    "promo": p.get("active_promotion"),
                    "desc": (p.get("description") or "")[:140],
                    "note": (p.get("data_quality_notes") or "")[:120] or None,
                }
            )
        compact.update(
            {
                "total_available": result.get("total_available_matching"),
                "products": products,
                "alternatives": [
                    {
                        "id": a.get("product_id"),
                        "name": a.get("name"),
                        "price": a.get("effective_price_brl"),
                    }
                    for a in (result.get("suggested_alternatives_for_unavailable") or [])[:3]
                ],
                "out_of_scope": result.get("status") == "out_of_scope_accessory",
                "policy_rule": result.get("policy_rule"),
                "message": result.get("message"),
            }
        )
    elif "brands" in result:
        compact.update(
            {
                "brands_count": result.get("brands_count"),
                "category": result.get("resolved_category"),
                "brands": [
                    {
                        "brand": b.get("brand"),
                        "count": b.get("product_count"),
                        "min": b.get("min_price_brl"),
                        "max": b.get("max_price_brl"),
                    }
                    for b in (result.get("brands") or [])[:20]
                ],
            }
        )
    elif "orders" in result:
        compact["orders"] = []
        for o in (result.get("orders") or [])[:3]:
            compact["orders"].append(
                {
                    "order_id": o.get("order_id"),
                    "status": o.get("status_pt") or o.get("status"),
                    "date": o.get("order_date"),
                    "delivery": o.get("estimated_or_actual_delivery"),
                    "total": o.get("total_brl"),
                    "payment": o.get("payment_method"),
                    "tracking": o.get("tracking_code"),
                    "customer": (o.get("customer") or {}).get("name"),
                    "city": (o.get("customer") or {}).get("city"),
                    "items": [
                        {
                            "name": i.get("product_name"),
                            "qty": i.get("quantity"),
                            "price": i.get("unit_price_brl"),
                        }
                        for i in (o.get("items") or [])[:5]
                    ],
                    "eligibility": o.get("return_and_warranty_eligibility"),
                }
            )
        compact["reference_date"] = result.get("reference_date_used")
        compact["message"] = result.get("message")
    elif "retrieved_sections" in result:
        compact["sections"] = [
            {
                "section": s.get("section_number"),
                "title": s.get("section_title"),
                "content": (s.get("content") or "")[:700],
            }
            for s in (result.get("retrieved_sections") or [])[:2]
        ]
        compact["store"] = {
            "name": (result.get("store_metadata") or {}).get("nome_fantasia"),
            "address": (result.get("store_metadata") or {}).get("endereco"),
            "hours": {
                "seg_sex": (result.get("store_metadata") or {}).get("horario_seg_sex"),
                "sab": (result.get("store_metadata") or {}).get("horario_sabado"),
            },
        }
    elif "active_promotions" in result:
        compact["active"] = (result.get("active_promotions") or [])[:8]
        compact["expired_sample"] = (result.get("expired_promotions") or [])[:3]
        compact["policy_rules"] = (result.get("policy_rules") or "")[:280]
    elif "payment_options" in result:
        compact = {
            "amount": result.get("base_amount_brl"),
            "payment": result.get("payment_options"),
            "shipping": result.get("shipping_options"),
        }
    else:
        compact = result

    raw = json.dumps(compact, ensure_ascii=False, default=str)
    if len(raw) > MAX_TOOL_RESULT_CHARS:
        return {"truncated": True, "data": raw[:MAX_TOOL_RESULT_CHARS]}
    return compact
