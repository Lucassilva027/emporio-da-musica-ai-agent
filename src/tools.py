"""Ferramentas determinísticas para Function Calling / ReAct do Agente da Empório da Música.

Conecta o LLM ao banco SQLite operacional (catálogo, estoque, promoções, clientes, pedidos)
e ao motor RAG híbrido de políticas internas (políticas_da_loja.pdf).
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from typing import Any

from src.config import STORE_INFO, get_reference_date
from src.database import StoreDatabase, normalize_text
from src.rag import PolicyRAGEngine

ACCESSORY_KEYWORDS = [
    "encordoamento",
    "jogo de cordas",
    "corda avulsa",
    "cordas avulsas",
    "palheta",
    "palhetas",
    "cabo",
    "cabos",
    "case",
    "cases",
    "capa",
    "bag",
    "pedal",
    "pedais",
    "pedaleira",
    "amplificador",
    "amplificadores",
    "cubo",
    "suporte",
    "estante",
    "correia",
    "capotraste",
    "baqueta",
    "baquetas",
    "afinador digital",
]

CATEGORY_ALIASES: dict[str, list[str]] = {
    "Violões": ["violao", "violoes", "acustico", "dreadnought", "folk", "classico", "7 cordas"],
    "Ukuleles": ["ukulele", "ukuleles", "soprano", "concert", "tenor", "baritono"],
    "Guitarras": ["guitarra", "guitarras", "stratocaster", "telecaster", "les paul", "rg550", "prs"],
    "Baixos": ["baixo", "baixos", "contrabaixo", "jazz bass", "precision", "stingray"],
    "Baterias e Percussão": ["bateria", "baterias", "percussao", "bumbo", "tambor"],
    "Teclados e Pianos": ["teclado", "teclados", "piano", "pianos", "sintetizador", "synth", "arranjador", "midi"],
    "Instrumentos de Sopro (Madeiras)": ["sopro", "flauta", "clarinete", "saxofone", "sax", "oboe", "madeira"],
    "Instrumentos de Sopro (Metais)": ["trompete", "trombone", "tuba", "trompa", "metais"],
    "Cordas Orquestrais": ["violino", "viola de arco", "violoncelo", "cello", "orquestral", "orquestrais"],
}


def compute_installment_plan(price_brl: float) -> dict[str, Any]:
    """Calcula o parcelamento máximo sem juros respeitando as regras da Seção 3.1 do Manual:
    - Até 3x sem juros: parcela mínima de R$ 50,00
    - De 4x a 6x sem juros: parcela mínima de R$ 80,00
    - De 7x a 12x sem juros: parcela mínima de R$ 100,00
    """
    best_n = 1
    for n in range(12, 0, -1):
        installment_val = price_brl / n
        if 7 <= n <= 12 and installment_val >= 100.0 - 1e-6:
            best_n = n
            break
        if 4 <= n <= 6 and installment_val >= 80.0 - 1e-6:
            best_n = max(best_n, n)
            break
        if 1 <= n <= 3 and (n == 1 or installment_val >= 50.0 - 1e-6):
            best_n = max(best_n, n)
            break

    return {
        "max_installments": best_n,
        "installment_value_brl": round(price_brl / best_n, 2),
        "summary": f"até {best_n}x de R$ {price_brl / best_n:,.2f} sem juros".replace(",", "X").replace(".", ",").replace("X", "."),
    }


def enrich_product_row(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Enriquece um registro de produto com promoções ativas/expiradas, preço PIX e parcelamento."""
    pid = row["product_id"]
    base_price = float(row["price_brl"])

    promos = conn.execute(
        """
        SELECT promotion_id, discount_percent, description, is_active
        FROM promotions
        WHERE product_id = ?
        ORDER BY is_active DESC, discount_percent DESC
        """,
        (pid,),
    ).fetchall()

    active_promo = None
    expired_promos = []
    for p in promos:
        p_dict = dict(p)
        disc = float(p_dict["discount_percent"])
        promo_price = round(base_price * (1.0 - disc / 100.0), 2)
        p_info = {
            "promotion_id": p_dict["promotion_id"],
            "campaign": p_dict["description"],
            "discount_percent": disc,
            "promotional_price_brl": promo_price,
        }
        if p_dict["is_active"] == 1 and active_promo is None:
            active_promo = p_info
        else:
            expired_promos.append(p_info)

    # Regra 6.2: Promoções não são cumulativas. O desconto de PIX (5%) não se aplica sobre preços já promocionais.
    if active_promo is not None:
        effective_price = active_promo["promotional_price_brl"]
        pix_price = effective_price
        pix_note = (
            f"R$ {pix_price:.2f} (preço promocional com {active_promo['discount_percent']:.0f}% OFF da campanha "
            f"'{active_promo['campaign']}'; desconto de 5% do PIX não cumulativo conforme Seção 6.2)"
        )
    else:
        effective_price = base_price
        pix_price = round(base_price * 0.95, 2)
        pix_note = f"R$ {pix_price:.2f} (5% de desconto à vista no PIX sobre R$ {base_price:.2f})"

    installments = compute_installment_plan(effective_price)

    try:
        specs = json.loads(row["specs_json"]) if row.get("specs_json") else {}
    except Exception:
        specs = {}

    return {
        "product_id": pid,
        "name": row["name"],
        "category_id": row["category_id"],
        "category_name": row["category_name"],
        "original_price_brl": base_price,
        "effective_price_brl": effective_price,
        "pix_price_brl": pix_price,
        "pix_rule_explanation": pix_note,
        "installment_plan": installments,
        "stock_quantity": row["stock_quantity"],
        "raw_status": row["raw_status"],
        "effective_status": row["effective_status"],
        "is_available": bool(row["is_available"]),
        "strings_type": row["strings_type"],
        "has_electronics": bool(row["has_electronics"]),
        "color": row["color"],
        "description": row["description"],
        "specs": specs,
        "active_promotion": active_promo,
        "expired_promotions": expired_promos,
        "data_quality_notes": row["data_quality_notes"],
    }


class StoreToolkit:
    """Conjunto de ferramentas disponíveis para o agente consultar dados e políticas."""

    def __init__(
        self,
        db: StoreDatabase | None = None,
        rag: PolicyRAGEngine | None = None,
    ) -> None:
        self.db = db or StoreDatabase()
        self.rag = rag or PolicyRAGEngine()

    def search_products(
        self,
        query: str = "",
        category: str = "",
        max_price: float | None = None,
        min_price: float | None = None,
        strings_type: str = "",
        has_electronics: bool | None = None,
        only_available: bool = False,
        only_promotions: bool = False,
        limit: int = 12,
    ) -> dict[str, Any]:
        """Busca instrumentos musicais no catálogo por nome, categoria, faixa de preço e especificações."""
        q_norm = normalize_text(query)
        cat_norm = normalize_text(category)

        # Verifica se o cliente está buscando acessórios (fora de escopo da loja conforme Seção 1 e 7.1)
        combined_input = f"{q_norm} {cat_norm}".strip()
        matched_accessories = [
            acc for acc in ACCESSORY_KEYWORDS if normalize_text(acc) in combined_input
        ]
        # Exceção: "violao 7 cordas" ou "violao cordas de nylon/aco" é instrumento, não acessório
        is_instrument_mention = any(
            w in combined_input
            for w in ["violao", "violoes", "ukulele", "guitarra", "baixo", "bateria", "teclado", "piano"]
        )
        if matched_accessories and not is_instrument_mention:
            return {
                "status": "out_of_scope_accessory",
                "matched_accessory_terms": matched_accessories,
                "policy_rule": (
                    "A Empório da Música trabalha exclusivamente com instrumentos musicais. "
                    "Não comercializamos acessórios como cordas avulsas, palhetas, cabos, cases, "
                    "pedais ou amplificadores (Seções 1 e 7.1 do Manual). Redirecione educadamente "
                    "o cliente e sugira procurar lojas parceiras especializadas em acessórios."
                ),
                "products": [],
            }

        # Resolve categoria caso tenha sido passada na query ou em category
        resolved_category: str | None = None
        for official_cat, aliases in CATEGORY_ALIASES.items():
            if cat_norm and (cat_norm in normalize_text(official_cat) or any(a in cat_norm for a in aliases)):
                resolved_category = official_cat
                break
        if not resolved_category and q_norm:
            for official_cat, aliases in CATEGORY_ALIASES.items():
                if any(re.search(rf"\b{re.escape(a)}\b", q_norm) for a in aliases):
                    resolved_category = official_cat
                    break

        with self.db.get_connection() as conn:
            sql = "SELECT * FROM products WHERE 1=1"
            params: list[Any] = []

            if resolved_category:
                sql += " AND category_name = ?"
                params.append(resolved_category)

            if strings_type:
                st_norm = normalize_text(strings_type)
                if "nylon" in st_norm:
                    sql += " AND strings_type = 'nylon'"
                elif "aco" in st_norm or "steel" in st_norm:
                    sql += " AND strings_type = 'aco'"

            if has_electronics is not None:
                sql += " AND has_electronics = ?"
                params.append(1 if has_electronics else 0)

            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

            # Se a categoria existe mas tem 0 produtos no banco (ex: Sopro ou Cordas Orquestrais)
            if resolved_category and not rows:
                return {
                    "status": "empty_category",
                    "category": resolved_category,
                    "message": (
                        f"A categoria '{resolved_category}' faz parte do cadastro da Empório da Música, "
                        "porém no momento não há nenhum instrumento dessa categoria disponível em estoque no sistema."
                    ),
                    "products": [],
                }

            # Filtra por termos de busca (ex: "Takamine GD20", "Yamaha", "7 cordas", "baritono")
            stop_words = {
                "violao", "violoes", "ukulele", "ukuleles", "guitarra", "guitarras",
                "baixo", "baixos", "contrabaixo", "bateria", "baterias", "teclado",
                "teclados", "piano", "sintetizador", "de", "da", "do", "dos", "das",
                "com", "para", "ate", "disponivel", "disponiveis", "opcoes", "opcao",
                "quero", "tem", "qual", "quais", "preco", "valor", "custa", "custando",
                "cordas", "instrumento", "instrumentos", "reais", "rs",
            }
            # Remove tokens monetários / ruído numérico genérico da query livre
            query_tokens = []
            for t in q_norm.split():
                if len(t) < 2 or t in stop_words:
                    continue
                if re.fullmatch(r"r\$?\d+(?:[.,]\d+)?", t) or re.fullmatch(r"\d+(?:[.,]\d+)?", t):
                    continue
                query_tokens.append(t)
            # Mantém "7" se for busca de 7 cordas
            if "7 cordas" in q_norm and "7" not in query_tokens:
                query_tokens.append("7")

            if query_tokens:
                scored_rows: list[tuple[int, dict[str, Any]]] = []
                for r in rows:
                    hits = sum(1 for tok in query_tokens if tok in r["search_text"])
                    # Boost para match no nome do produto
                    name_hits = sum(2 for tok in query_tokens if tok in r["name_norm"])
                    if hits > 0:
                        scored_rows.append((hits + name_hits, r))
                if scored_rows:
                    scored_rows.sort(key=lambda x: (-x[0], x[1]["price_brl"]))
                    rows = [r for _, r in scored_rows]
                elif not resolved_category:
                    # Sem categoria resolvida e sem match lexical → lista vazia
                    rows = []
                # Com categoria resolvida e tokens sem match (ex.: "custando"),
                # mantém o recorte por categoria/preço já aplicado.

            enriched = [enrich_product_row(r, conn) for r in rows]

            # Aplica filtros de preço considerando tanto o preço de tabela quanto o preço efetivo (promocional)
            if max_price is not None:
                enriched = [
                    p
                    for p in enriched
                    if p["effective_price_brl"] <= max_price or p["original_price_brl"] <= max_price
                ]
            if min_price is not None:
                enriched = [
                    p
                    for p in enriched
                    if p["effective_price_brl"] >= min_price or p["original_price_brl"] >= min_price
                ]

            if only_promotions:
                enriched = [p for p in enriched if p["active_promotion"] is not None]

            # Separa disponíveis de indisponíveis/descontinuados/em breve
            available_items = [p for p in enriched if p["is_available"]]
            unavailable_items = [p for p in enriched if not p["is_available"]]

            if only_available:
                final_items = sorted(available_items, key=lambda x: x["effective_price_brl"])[:limit]
            else:
                # Se a busca foi específica (poucos itens), mostra disponíveis e indisponíveis com alternativas
                if not query_tokens:
                    final_items = sorted(available_items, key=lambda x: x["effective_price_brl"])[:limit]
                else:
                    final_items = enriched[:limit]

            # Se algum item retornado está indisponível/descontinuado (Seção 7.3), sugere alternativas disponíveis
            alternatives: list[dict[str, Any]] = []
            if unavailable_items and len(final_items) <= 3:
                target = unavailable_items[0]
                alt_rows = conn.execute(
                    """
                    SELECT * FROM products
                    WHERE category_id = ? AND is_available = 1 AND product_id != ?
                    ORDER BY ABS(price_brl - ?) ASC
                    LIMIT 3
                    """,
                    (target["category_id"], target["product_id"], target["original_price_brl"]),
                ).fetchall()
                alternatives = [enrich_product_row(dict(ar), conn) for ar in alt_rows]

            return {
                "status": "ok",
                "total_available_matching": len(available_items),
                "total_unavailable_matching": len(unavailable_items),
                "applied_filters": {
                    "query": query,
                    "resolved_category": resolved_category,
                    "max_price": max_price,
                    "min_price": min_price,
                    "strings_type": strings_type,
                    "has_electronics": has_electronics,
                    "only_available": only_available,
                    "only_promotions": only_promotions,
                },
                "products": final_items,
                "unavailable_or_discontinued_matches": unavailable_items[:3],
                "suggested_alternatives_for_unavailable": alternatives,
            }

    def get_promotions(
        self,
        include_expired: bool = False,
        campaign_name: str = "",
    ) -> dict[str, Any]:
        """Lista promoções ativas no sistema e, opcionalmente, promoções expiradas para transparência."""
        camp_norm = normalize_text(campaign_name)
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT pr.promotion_id, pr.product_id, pr.discount_percent, pr.description AS campaign,
                       pr.is_active, p.*
                FROM promotions pr
                JOIN products p ON pr.product_id = p.product_id
                ORDER BY pr.is_active DESC, pr.discount_percent DESC
                """
            ).fetchall()

            active_list = []
            expired_list = []
            for r in rows:
                r_dict = dict(r)
                if camp_norm and camp_norm not in normalize_text(r_dict["campaign"]):
                    continue
                enriched_prod = enrich_product_row(r_dict, conn)
                disc = float(r_dict["discount_percent"])
                orig = float(r_dict["price_brl"])
                promo_val = round(orig * (1.0 - disc / 100.0), 2)
                entry = {
                    "promotion_id": r_dict["promotion_id"],
                    "campaign": r_dict["campaign"],
                    "is_active": bool(r_dict["is_active"]),
                    "product_id": r_dict["product_id"],
                    "product_name": r_dict["name"],
                    "category_name": r_dict["category_name"],
                    "original_price_brl": orig,
                    "discount_percent": disc,
                    "promotional_price_brl": promo_val,
                    "stock_quantity": r_dict["stock_quantity"],
                    "effective_status": r_dict["effective_status"],
                    "installment_plan": enriched_prod["installment_plan"],
                }
                if r_dict["is_active"] == 1:
                    active_list.append(entry)
                elif include_expired or camp_norm:
                    expired_list.append(entry)

            return {
                "active_promotions_count": len(active_list),
                "active_promotions": active_list,
                "expired_promotions": expired_list[:10],
                "policy_rules": (
                    "Seção 6.2 e 7.3: Promoções não são cumulativas (desconto de 5% no PIX não acumula "
                    "com preço promocional). Sempre apresentar preço original, percentual de desconto e "
                    "preço promocional. Se o cliente perguntar sobre promoção expirada, informar com "
                    "transparência que está encerrada e apresentar o preço vigente."
                ),
            }

    def check_order_status(
        self,
        order_id: int | None = None,
        customer_name: str = "",
        customer_email: str = "",
        customer_phone: str = "",
    ) -> dict[str, Any]:
        """Consulta status de pedidos, rastreamento e avalia automaticamente elegibilidade de devolução/troca/garantia."""
        ref_date = get_reference_date()
        name_norm = normalize_text(customer_name)
        email_norm = customer_email.strip().lower()
        phone_digits = "".join(ch for ch in customer_phone if ch.isdigit())

        with self.db.get_connection() as conn:
            matched_orders: list[dict[str, Any]] = []

            if order_id is not None:
                rows = conn.execute(
                    """
                    SELECT o.*, c.name AS customer_name, c.email AS customer_email,
                           c.phone AS customer_phone, c.city AS customer_city
                    FROM orders o
                    JOIN customers c ON o.customer_id = c.customer_id
                    WHERE o.order_id = ?
                    """,
                    (int(order_id),),
                ).fetchall()
                matched_orders = [dict(r) for r in rows]
            else:
                # Busca cliente por email, telefone ou nome
                cust_sql = "SELECT * FROM customers WHERE 1=1"
                cust_params: list[Any] = []
                if email_norm:
                    cust_sql += " AND email_norm = ?"
                    cust_params.append(email_norm)
                if phone_digits:
                    cust_sql += " AND phone_digits LIKE ?"
                    cust_params.append(f"%{phone_digits}%")
                if name_norm:
                    cust_sql += " AND name_norm LIKE ?"
                    cust_params.append(f"%{name_norm}%")

                if not (email_norm or phone_digits or name_norm):
                    return {
                        "status": "missing_identifier",
                        "message": (
                            "Para consultar um pedido com segurança (LGPD - Seção 9), solicite ao cliente "
                            "o número do pedido (ex.: #1 a #20), nome completo, e-mail ou telefone."
                        ),
                    }

                customers = [dict(c) for c in conn.execute(cust_sql, cust_params).fetchall()]
                if not customers:
                    return {
                        "status": "customer_not_found",
                        "message": "Nenhum cliente ou pedido encontrado com os dados informados.",
                    }

                # Se encontrou múltiplos clientes distintos (homônimos como 'Bruno', 'Diego', 'Thiago', 'Camila'),
                # evita vazar dados de terceiros (LGPD) e pede confirmação de sobrenome, e-mail ou nº do pedido.
                if len(customers) > 1 and not email_norm and not phone_digits:
                    exact_match = [c for c in customers if c["name_norm"] == name_norm]
                    if len(exact_match) == 1:
                        customers = exact_match
                    else:
                        return {
                            "status": "ambiguous_customer_lgpd",
                            "matches_count": len(customers),
                            "message": (
                                f"Encontramos {len(customers)} clientes com o nome '{customer_name}'. "
                                "Por segurança e conformidade com a LGPD (Seção 9), peça o nome completo, "
                                "e-mail, telefone ou o número do pedido antes de exibir detalhes."
                            ),
                        }

                cust_ids = [c["customer_id"] for c in customers]
                placeholders = ",".join("?" for _ in cust_ids)
                rows = conn.execute(
                    f"""
                    SELECT o.*, c.name AS customer_name, c.email AS customer_email,
                           c.phone AS customer_phone, c.city AS customer_city
                    FROM orders o
                    JOIN customers c ON o.customer_id = c.customer_id
                    WHERE o.customer_id IN ({placeholders})
                    ORDER BY o.order_date DESC
                    """,
                    cust_ids,
                ).fetchall()
                matched_orders = [dict(r) for r in rows]

                if not matched_orders:
                    c0 = customers[0]
                    return {
                        "status": "no_orders_for_customer",
                        "customer": {
                            "customer_id": c0["customer_id"],
                            "name": c0["name"],
                            "city": c0["city"],
                        },
                        "message": f"O cliente {c0['name']} está cadastrado, mas não possui pedidos registrados no momento.",
                    }

            if not matched_orders:
                return {
                    "status": "order_not_found",
                    "message": f"Pedido #{order_id} não encontrado no sistema.",
                }

            detailed_orders = []
            for o in matched_orders:
                oid = o["order_id"]
                items_rows = conn.execute(
                    """
                    SELECT oi.quantity, oi.unit_price_brl, oi.line_total_brl,
                           p.product_id, p.name AS product_name, p.category_name, p.raw_status
                    FROM order_items oi
                    JOIN products p ON oi.product_id = p.product_id
                    WHERE oi.order_id = ?
                    """,
                    (oid,),
                ).fetchall()
                items = [dict(ir) for ir in items_rows]

                # Avalia prazos de troca, devolução e garantia em relação à data de referência
                order_dt = date.fromisoformat(o["order_date"])
                days_since_order = (ref_date - order_dt).days

                deliv_dt = (
                    date.fromisoformat(o["estimated_delivery"])
                    if o.get("estimated_delivery")
                    else None
                )
                days_since_delivery = (
                    (ref_date - deliv_dt).days if deliv_dt and o["status"] == "delivered" else None
                )

                policy_eval = self._evaluate_order_policies(
                    status=o["status"],
                    days_since_order=days_since_order,
                    days_since_delivery=days_since_delivery,
                    ref_date=ref_date,
                    order_date=o["order_date"],
                    delivery_date=o.get("estimated_delivery"),
                )

                status_pt = {
                    "delivered": "Entregue",
                    "shipped": "Enviado / Em trânsito",
                    "confirmed": "Confirmado (em separação)",
                    "pending": "Pendente (aguardando confirmação de pagamento/processamento)",
                    "cancelled": "Cancelado",
                }.get(o["status"], o["status"])

                detailed_orders.append(
                    {
                        "order_id": oid,
                        "status": o["status"],
                        "status_pt": status_pt,
                        "order_date": o["order_date"],
                        "estimated_or_actual_delivery": o["estimated_delivery"],
                        "total_brl": o["total_brl"],
                        "items_subtotal_brl": o["items_subtotal_brl"],
                        "payment_method": o["payment_method"],
                        "tracking_code": o["tracking_code"],
                        "notes": o["notes"],
                        "customer": {
                            "customer_id": o["customer_id"],
                            "name": o["customer_name"],
                            "city": o["customer_city"],
                            "email": o["customer_email"],
                            "phone": o["customer_phone"],
                        },
                        "items": items,
                        "return_and_warranty_eligibility": policy_eval,
                    }
                )

            return {
                "status": "ok",
                "reference_date_used": ref_date.isoformat(),
                "orders": detailed_orders,
            }

    def _evaluate_order_policies(
        self,
        status: str,
        days_since_order: int,
        days_since_delivery: int | None,
        ref_date: date,
        order_date: str,
        delivery_date: str | None,
    ) -> dict[str, Any]:
        """Aplica as regras da Seção 4 (Trocas e Devoluções) e Seção 8 (Garantia) a um pedido."""
        if status == "cancelled":
            return {
                "can_cancel_before_shipping": False,
                "regret_return_7_days": False,
                "preference_exchange_7_days": False,
                "defect_store_exchange_30_days": False,
                "legal_warranty_90_days": False,
                "explanation": "Pedido já se encontra cancelado.",
            }

        if status in ("pending", "confirmed"):
            return {
                "can_cancel_before_shipping": True,
                "regret_return_7_days": True,
                "preference_exchange_7_days": True,
                "defect_store_exchange_30_days": True,
                "legal_warranty_90_days": True,
                "explanation": (
                    f"O pedido está com status '{status}' e ainda não foi entregue. "
                    "O cliente pode solicitar cancelamento ou alteração imediatamente, e o prazo de "
                    "7 dias corridos de Direito de Arrependimento (Seção 4.1) passa a contar a partir do recebimento."
                ),
            }

        if status == "shipped":
            return {
                "can_cancel_before_shipping": False,
                "regret_return_7_days": True,
                "preference_exchange_7_days": True,
                "defect_store_exchange_30_days": days_since_order <= 30,
                "legal_warranty_90_days": True,
                "explanation": (
                    f"O pedido está em trânsito (previsão/entrega: {delivery_date}). "
                    "O cliente poderá exercer o Direito de Arrependimento em até 7 dias corridos após o recebimento "
                    "(Seção 4.1), com frete de devolução por conta da loja e reembolso em até 10 dias úteis."
                ),
            }

        # Pedido entregue (status == 'delivered')
        d_deliv = days_since_delivery if days_since_delivery is not None else days_since_order
        regret_ok = d_deliv <= 7
        defect_store_ok = days_since_order <= 30
        legal_warranty_ok = d_deliv <= 90

        return {
            "reference_date": ref_date.isoformat(),
            "days_since_order": days_since_order,
            "days_since_delivery": d_deliv,
            "regret_return_7_days": regret_ok,
            "preference_exchange_7_days": regret_ok,
            "defect_store_exchange_30_days": defect_store_ok,
            "legal_warranty_90_days": legal_warranty_ok,
            "manufacturer_warranty_6m_to_2y": True,
            "explanation": (
                f"Pedido comprado em {order_date} ({days_since_order} dias atrás) e entregue em {delivery_date} "
                f"({d_deliv} dias atrás, considerando a data de referência {ref_date.isoformat()}). "
                f"- Direito de Arrependimento / Troca por Preferência (7 dias corridos do recebimento): "
                f"{'DENTRO DO PRAZO' if regret_ok else f'FORA DO PRAZO ({d_deliv} dias corridos desde a entrega; limite é 7 dias)'}. "
                f"- Troca direta na loja por defeito de fabricação (30 dias corridos da compra): "
                f"{'DENTRO DO PRAZO' if defect_store_ok else f'FORA DO PRAZO ({days_since_order} dias desde a compra; limite na loja é 30 dias, acionar garantia do fabricante com intermediação da loja)'}. "
                f"- Garantia Legal CDC (90 dias do recebimento): "
                f"{'DENTRO DO PRAZO' if legal_warranty_ok else 'FORA DO PRAZO DE 90 DIAS (verificar garantia contratual do fabricante de 6 meses a 2 anos)'}."
            ),
        }

    def calculate_payment_and_shipping(
        self,
        amount_brl: float,
        city: str = "Campo Grande",
        has_active_promotion: bool = False,
        is_large_instrument: bool = False,
    ) -> dict[str, Any]:
        """Calcula opções de pagamento, parcelamento sem juros e regras de frete para um valor e cidade."""
        city_norm = normalize_text(city)
        is_cg = city_norm in ("campo grande", "cg", "campo grande ms", "")

        # Pagamento (Seção 3 e 6.2)
        if has_active_promotion:
            pix_val = round(amount_brl, 2)
            pix_desc = (
                f"R$ {pix_val:.2f} à vista no PIX (preço já promocional; desconto de 5% do PIX "
                "não é cumulativo com promoções conforme Seção 6.2)"
            )
        else:
            pix_val = round(amount_brl * 0.95, 2)
            pix_desc = f"R$ {pix_val:.2f} à vista no PIX (5% de desconto sobre R$ {amount_brl:.2f})"

        installments = compute_installment_plan(amount_brl)
        all_brackets = []
        for n in range(1, 13):
            val = amount_brl / n
            if n == 1 or (n <= 3 and val >= 50) or (4 <= n <= 6 and val >= 80) or (7 <= n <= 12 and val >= 100):
                all_brackets.append({"installments": n, "value_brl": round(val, 2)})

        # Frete (Seção 5.1 e 5.2)
        if is_cg:
            shipping_cost = 0.0 if amount_brl >= 500.0 else 35.0
            shipping_info = {
                "region": "Região Metropolitana de Campo Grande/MS",
                "shipping_cost_brl": shipping_cost,
                "is_free_shipping": shipping_cost == 0.0,
                "delivery_method": "Motoboy próprio (contato telefônico prévio antes da entrega)",
                "estimated_days": "1 a 3 dias úteis",
                "rule": "Frete grátis para pedidos acima de R$ 500,00; taxa fixa de R$ 35,00 abaixo de R$ 500,00 (Seção 5.1).",
            }
        else:
            shipping_info = {
                "region": f"Outras Cidades ({city})",
                "carriers": [
                    {"modality": "PAC (Correios)", "estimated_days": "5 a 12 dias úteis", "tracking": True, "insurance": True},
                    {"modality": "SEDEX (Correios)", "estimated_days": "2 a 5 dias úteis", "tracking": True, "insurance": True},
                    {"modality": "Jadlog (.package)", "estimated_days": "3 a 8 dias úteis", "tracking": True, "insurance": True},
                ],
                "large_instrument_warning": (
                    "Instrumentos de grande porte (baterias acústicas, pianos digitais, contrabaixos) "
                    "podem exigir frete especial com cotação individual."
                    if is_large_instrument
                    else None
                ),
                "rule": "Valor calculado com base no CEP, peso e dimensões. Todos os envios possuem seguro incluso (Seção 5.2).",
            }

        return {
            "base_amount_brl": round(amount_brl, 2),
            "has_active_promotion": has_active_promotion,
            "payment_options": {
                "pix": {"amount_brl": pix_val, "description": pix_desc},
                "debit_card": {"amount_brl": round(amount_brl, 2), "description": "Pagamento à vista (todas as bandeiras)"},
                "boleto": {"amount_brl": round(amount_brl, 2), "description": "Pagamento à vista (compensação em até 3 dias úteis)"},
                "credit_card": {
                    "max_installments": installments["max_installments"],
                    "installment_value_brl": installments["installment_value_brl"],
                    "summary": installments["summary"],
                    "valid_installment_options": all_brackets,
                },
                "split_payment_allowed": amount_brl > 2000.0,
                "split_payment_rule": (
                    "Permitida combinação de formas de pagamento (ex.: PIX + cartão) pois a compra supera R$ 2.000,00 (Seção 3.1)."
                    if amount_brl > 2000.0
                    else "Combinação de formas de pagamento (ex.: PIX + cartão) só é permitida para compras acima de R$ 2.000,00 (Seção 3.1)."
                ),
            },
            "shipping_options": shipping_info,
        }

    def search_store_policies(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """Busca trechos relevantes no Manual de Políticas e Procedimentos da Empório da Música (PDF)."""
        results = self.rag.search(query=query, top_k=top_k)
        return {
            "query": query,
            "store_metadata": STORE_INFO,
            "retrieved_sections": results,
        }

    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Retorna os schemas JSON compatíveis com OpenAI / Groq / Gemini / Ollama Function Calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": (
                        "Consulta o catálogo de instrumentos musicais da Empório da Música no banco de dados. "
                        "Use SEMPRE que o cliente perguntar sobre produtos, modelos, marcas, preços, "
                        "estoque/disponibilidade, especificações técnicas ou opções dentro de uma faixa de preço."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Termo livre de busca (ex.: 'Takamine GD20', 'Yamaha', '7 cordas', 'Stratocaster', 'barítono').",
                            },
                            "category": {
                                "type": "string",
                                "description": "Categoria do instrumento (ex.: 'Violões', 'Ukuleles', 'Guitarras', 'Baixos', 'Baterias e Percussão', 'Teclados e Pianos', 'Sopro', 'Cordas Orquestrais').",
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Preço máximo em reais (R$). Ex.: 1000 para 'até R$ 1000'.",
                            },
                            "min_price": {
                                "type": "number",
                                "description": "Preço mínimo em reais (R$).",
                            },
                            "strings_type": {
                                "type": "string",
                                "description": "Filtro de tipo de encordoamento para violões: 'nylon' ou 'aco'.",
                            },
                            "has_electronics": {
                                "type": "boolean",
                                "description": "Se true, filtra apenas instrumentos elétricos/eletroacústicos (com captação).",
                            },
                            "only_available": {
                                "type": "boolean",
                                "description": "Se true, retorna apenas produtos disponíveis em pronta-entrega (estoque > 0 e status active).",
                            },
                            "only_promotions": {
                                "type": "boolean",
                                "description": "Se true, retorna apenas produtos com promoção ativa no momento.",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_promotions",
                    "description": (
                        "Lista as promoções e descontos cadastrados no sistema (ativos e, opcionalmente, expirados). "
                        "Use quando o cliente perguntar sobre ofertas, descontos, queima de estoque, Black Friday, "
                        "Aniversário da Loja ou Semana do Músico."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_expired": {
                                "type": "boolean",
                                "description": "Se true, inclui promoções expiradas (is_active=0) para informar ao cliente caso ele pergunte sobre uma promoção encerrada.",
                            },
                            "campaign_name": {
                                "type": "string",
                                "description": "Nome da campanha (ex.: 'Black Friday', 'Desconto Exclusivo', 'Oferta Especial', 'Semana do Músico').",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_order_status",
                    "description": (
                        "Consulta detalhes e status de pedidos de clientes, código de rastreio e calcula "
                        "automaticamente se o pedido está dentro do prazo de Direito de Arrependimento (7 dias), "
                        "Troca por Defeito (30 dias) e Garantia Legal (90 dias)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "integer",
                                "description": "Número identificador do pedido (ex.: 1 a 20).",
                            },
                            "customer_name": {
                                "type": "string",
                                "description": "Nome do cliente para buscar seus pedidos.",
                            },
                            "customer_email": {
                                "type": "string",
                                "description": "E-mail do cliente.",
                            },
                            "customer_phone": {
                                "type": "string",
                                "description": "Telefone do cliente.",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_payment_and_shipping",
                    "description": (
                        "Calcula exatamente o valor no PIX, as opções válidas de parcelamento sem juros "
                        "(respeitando parcelas mínimas de R$ 50, R$ 80 e R$ 100) e o custo/prazo de frete "
                        "para Campo Grande ou outras cidades."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount_brl": {
                                "type": "number",
                                "description": "Valor total do produto ou pedido em R$.",
                            },
                            "city": {
                                "type": "string",
                                "description": "Cidade de destino para cálculo de frete (padrão: 'Campo Grande').",
                            },
                            "has_active_promotion": {
                                "type": "boolean",
                                "description": "True se o valor já for promocional (pois o desconto de 5% do PIX não acumula com promoções).",
                            },
                            "is_large_instrument": {
                                "type": "boolean",
                                "description": "True para baterias acústicas, pianos/teclados grandes ou contrabaixos.",
                            },
                        },
                        "required": ["amount_brl"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_store_policies",
                    "description": (
                        "Realiza busca semântica e léxica (RAG) no Manual de Políticas e Procedimentos da "
                        "Empório da Música (PDF). Use SEMPRE que o cliente perguntar sobre endereço, telefone, "
                        "horário de funcionamento, formas de pagamento, regras de parcelamento, trocas, "
                        "devoluções, arrependimento, frete, garantia, acessórios ou LGPD."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Pergunta ou tópico a buscar no manual de políticas.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Quantidade de seções do manual a recuperar (padrão: 3).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Executa uma ferramenta pelo nome com validação segura de argumentos."""
        try:
            if name == "search_products":
                return self.search_products(
                    query=str(arguments.get("query") or ""),
                    category=str(arguments.get("category") or ""),
                    max_price=float(arguments["max_price"]) if arguments.get("max_price") is not None else None,
                    min_price=float(arguments["min_price"]) if arguments.get("min_price") is not None else None,
                    strings_type=str(arguments.get("strings_type") or ""),
                    has_electronics=bool(arguments["has_electronics"]) if arguments.get("has_electronics") is not None else None,
                    only_available=bool(arguments.get("only_available", False)),
                    only_promotions=bool(arguments.get("only_promotions", False)),
                )
            if name == "get_promotions":
                return self.get_promotions(
                    include_expired=bool(arguments.get("include_expired", False)),
                    campaign_name=str(arguments.get("campaign_name") or ""),
                )
            if name == "check_order_status":
                oid = arguments.get("order_id")
                return self.check_order_status(
                    order_id=int(oid) if oid is not None and str(oid).strip() != "" else None,
                    customer_name=str(arguments.get("customer_name") or ""),
                    customer_email=str(arguments.get("customer_email") or ""),
                    customer_phone=str(arguments.get("customer_phone") or ""),
                )
            if name == "calculate_payment_and_shipping":
                return self.calculate_payment_and_shipping(
                    amount_brl=float(arguments.get("amount_brl", 0.0)),
                    city=str(arguments.get("city") or "Campo Grande"),
                    has_active_promotion=bool(arguments.get("has_active_promotion", False)),
                    is_large_instrument=bool(arguments.get("is_large_instrument", False)),
                )
            if name == "search_store_policies":
                return self.search_store_policies(
                    query=str(arguments.get("query") or ""),
                    top_k=int(arguments.get("top_k", 3)),
                )
            return {"error": f"Ferramenta desconhecida: {name}"}
        except Exception as exc:
            return {"error": f"Erro ao executar {name}: {exc}"}
