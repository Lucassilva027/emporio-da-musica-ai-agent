"""Testes de regressão das tools, RAG e motor local do agente."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import EmporioAgent
from src.database import StoreDatabase
from src.rag import PolicyRAGEngine
from src.tools import StoreToolkit, compute_installment_plan


def test_etl_loads_expected_counts():
    db = StoreDatabase()
    with db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 65
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 50
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 20
        assert conn.execute("SELECT COUNT(*) FROM promotions").fetchone()[0] == 25
        # Produto 96 active + stock 0 => out_of_stock
        row = conn.execute(
            "SELECT effective_status, is_available FROM products WHERE product_id = 96"
        ).fetchone()
        assert row["effective_status"] == "out_of_stock"
        assert row["is_available"] == 0


def test_rag_returns_return_policy_for_regret_query():
    rag = PolicyRAGEngine()
    hits = rag.search("Me arrependi da compra, posso devolver?")
    assert hits
    assert hits[0]["section_number"] == "4"


def test_search_violao_ate_1000():
    toolkit = StoreToolkit()
    result = toolkit.search_products(query="violões", category="Violões", max_price=1000, only_available=True)
    assert result["status"] == "ok"
    assert result["total_available_matching"] >= 1
    for p in result["products"]:
        assert p["effective_price_brl"] <= 1000 or p["original_price_brl"] <= 1000
        assert p["is_available"] is True
        assert p["category_name"] == "Violões"


def test_takamine_gd20_price():
    toolkit = StoreToolkit()
    result = toolkit.search_products(query="Takamine GD20")
    assert result["products"]
    p = result["products"][0]
    assert "Takamine GD20" in p["name"]
    assert abs(p["original_price_brl"] - 2199.0) < 0.01


def test_pix_not_cumulative_with_active_promo():
    toolkit = StoreToolkit()
    # product 127 has active promo (10%)
    result = toolkit.search_products(query="Kalani KAL-700T")
    p = next(x for x in result["products"] if x["product_id"] == 127)
    assert p["active_promotion"] is not None
    # PIX price must equal promotional price (no extra 5%)
    assert abs(p["pix_price_brl"] - p["effective_price_brl"]) < 0.01


def test_installment_rules():
    plan = compute_installment_plan(2199.0)
    # 2199/12 = 183.25 >= 100 => 12x ok
    assert plan["max_installments"] == 12
    plan2 = compute_installment_plan(200.0)
    # 200/4 = 50 < 80; 200/3 ≈ 66.67 >= 50 => max 3x
    assert plan2["max_installments"] == 3
    plan3 = compute_installment_plan(350.0)
    # 350/4 = 87.50 >= 80 => 4x
    assert plan3["max_installments"] == 4


def test_accessory_out_of_scope():
    toolkit = StoreToolkit()
    result = toolkit.search_products(query="palhetas")
    assert result["status"] == "out_of_scope_accessory"


def test_order_status_and_return_window():
    toolkit = StoreToolkit()
    # Pedido 1 entregue em 2025-10-25 — muito fora do prazo de 7 dias em 2026-03-23
    result = toolkit.check_order_status(order_id=1)
    assert result["status"] == "ok"
    elig = result["orders"][0]["return_and_warranty_eligibility"]
    assert elig["regret_return_7_days"] is False
    assert elig["legal_warranty_90_days"] is False

    # Pedido 8 shipped — ainda pode exercer arrependimento após recebimento
    result8 = toolkit.check_order_status(order_id=8)
    assert result8["orders"][0]["status"] == "shipped"
    assert result8["orders"][0]["return_and_warranty_eligibility"]["regret_return_7_days"] is True


def test_frete_campo_grande_rules():
    toolkit = StoreToolkit()
    cheap = toolkit.calculate_payment_and_shipping(400.0, city="Campo Grande")
    assert cheap["shipping_options"]["shipping_cost_brl"] == 35.0
    expensive = toolkit.calculate_payment_and_shipping(600.0, city="Campo Grande")
    assert expensive["shipping_options"]["is_free_shipping"] is True
    # PIX 5% sem promoção
    assert abs(expensive["payment_options"]["pix"]["amount_brl"] - 570.0) < 0.01


def test_list_brands_tool():
    toolkit = StoreToolkit()
    result = toolkit.list_brands(only_available=True)
    assert result["status"] == "ok"
    assert result["brands_count"] >= 5
    brand_names = {b["brand"] for b in result["brands"]}
    assert "Yamaha" in brand_names
    assert "Casio" not in brand_names

    teclados = toolkit.list_brands(category="Teclados e Pianos", only_available=True)
    teclado_brands = {b["brand"] for b in teclados["brands"]}
    assert teclado_brands.issubset({"Korg", "Nord", "Roland"})


def test_brl_markdown_escapes_dollar():
    from src.tools import format_brl

    assert r"\$" in format_brl(1127.0, markdown_safe=True)
    assert format_brl(1127.0, markdown_safe=False).startswith("R$")


def test_local_agent_scenarios():
    agent = EmporioAgent(provider="local")

    r1 = agent.chat("Quais opções de violões disponíveis custando até R$1000?")
    assert "viol" in r1.reply.lower() or "R\\$" in r1.reply or "R$" in r1.reply
    assert any(t["name"] == "search_products" for t in r1.tool_calls)
    # Streamlit-safe: preços não devem expor `$` sem escape
    assert "R$" not in r1.reply.replace(r"\$", "")

    r2 = agent.chat("Qual o endereço da loja?")
    assert "14 de Maio" in r2.reply or "Campo Grande" in r2.reply

    r3 = agent.chat("Quanto custa o Takamine GD20?")
    assert "2199" in r3.reply.replace(".", "").replace(",", "") or "2.199" in r3.reply

    r4 = agent.chat("Me arrependi da minha compra, posso devolver meu pedido #1?")
    assert any(t["name"] in ("search_store_policies", "check_order_status") for t in r4.tool_calls)
    assert "7 dias" in r4.reply.lower() or "arrependimento" in r4.reply.lower()

    r5 = agent.chat("Vocês vendem palhetas e cabos?")
    assert "exclusivamente" in r5.reply.lower() or "não vendemos" in r5.reply.lower() or "nao vendemos" in r5.reply.lower()

    r6 = agent.chat("Quais marcas vocês possuem?")
    assert any(t["name"] == "list_brands" for t in r6.tool_calls)
    assert "Yamaha" in r6.reply
    assert "marca" in r6.reply.lower()
