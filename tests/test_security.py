"""Testes de segurança, LGPD, allowlist de tools e isolamento de sessão."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import MAX_USER_MESSAGE_CHARS, EmporioAgent
from src.tools import ALLOWED_TOOLS, StoreToolkit, mask_email, mask_phone


def test_env_secrets_are_gitignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "storage/*.db" in gitignore


def test_tool_allowlist_rejects_unknown_and_injection_names():
    toolkit = StoreToolkit()
    for bad in [
        "drop_database",
        "search_products; DELETE FROM customers",
        "../../../etc/passwd",
        "os.system",
        "",
    ]:
        result = toolkit.execute_tool(bad, {})
        assert result["error"] == "tool_not_allowed"

    assert ALLOWED_TOOLS == {
        "search_products",
        "list_brands",
        "get_promotions",
        "check_order_status",
        "calculate_payment_and_shipping",
        "search_store_policies",
    }


def test_execute_tool_rejects_non_dict_arguments():
    toolkit = StoreToolkit()
    result = toolkit.execute_tool("search_products", "not-a-dict")  # type: ignore[arg-type]
    assert result["error"] == "invalid_arguments"


def test_sql_special_chars_in_product_query_do_not_break():
    toolkit = StoreToolkit()
    # Queries maliciosas/estranhas devem falhar com segurança (sem exception) graças a parâmetros
    payloads = [
        "Yamaha' OR '1'='1",
        'Takamine"; DROP TABLE products;--',
        "1 OR 1=1",
        "%" * 50,
    ]
    for payload in payloads:
        result = toolkit.execute_tool("search_products", {"query": payload})
        assert "error" not in result or result.get("error") != "execution_failed"
        assert "products" in result or result.get("status") == "out_of_scope_accessory"


def test_lgpd_ambiguous_customer_name_does_not_leak_orders():
    toolkit = StoreToolkit()
    # "Bruno" e "Camila" possuem homônimos na base
    result = toolkit.check_order_status(customer_name="Bruno")
    assert result["status"] == "ambiguous_customer_lgpd"
    assert "orders" not in result or not result.get("orders")
    assert "LGPD" in result["message"] or "lgpd" in result["message"].lower()


def test_order_response_masks_email_and_phone():
    toolkit = StoreToolkit()
    result = toolkit.check_order_status(order_id=1)
    assert result["status"] == "ok"
    cust = result["orders"][0]["customer"]
    assert "email" not in cust
    assert "phone" not in cust
    assert cust["email_masked"].endswith("@jmail.com")
    assert "***" in cust["email_masked"]
    assert "*****" in cust["phone_masked"]


def test_mask_helpers():
    assert mask_email("pedro.oliveira@jmail.com") == "p***@jmail.com"
    assert mask_phone("(67) 99812-3456").endswith("3456")
    assert "*****" in mask_phone("(67) 99812-3456")


def test_payment_amount_bounds():
    toolkit = StoreToolkit()
    assert toolkit.execute_tool(
        "calculate_payment_and_shipping", {"amount_brl": -10}
    )["error"] == "invalid_amount"
    assert toolkit.execute_tool(
        "calculate_payment_and_shipping", {"amount_brl": 9_999_999}
    )["error"] == "invalid_amount"


def test_policy_top_k_is_capped():
    toolkit = StoreToolkit()
    result = toolkit.execute_tool(
        "search_store_policies",
        {"query": "horário", "top_k": 999},
    )
    assert len(result["retrieved_sections"]) <= 5


def test_oversized_user_message_is_truncated_not_crashed():
    agent = EmporioAgent(provider="local")
    huge = "Quero um violão " + ("x" * (MAX_USER_MESSAGE_CHARS + 500))
    result = agent.chat(huge)
    assert result.reply
    assert len(result.reply) > 0


def test_session_isolation_between_agents():
    a1 = EmporioAgent(provider="local")
    a2 = EmporioAgent(provider="local")
    assert a1.session_id != a2.session_id
    a1.chat("Quanto custa o Takamine GD20?")
    msgs1 = a1.db.get_session_messages(a1.session_id)
    msgs2 = a2.db.get_session_messages(a2.session_id)
    assert any(m["role"] == "user" for m in msgs1)
    assert msgs2 == []


def test_out_of_scope_non_store_topic():
    agent = EmporioAgent(provider="local")
    result = agent.chat("Me passa uma receita de bolo de chocolate?")
    assert any(
        k in result.reply.lower()
        for k in ["instrumento", "empório", "emporio", "música", "musica"]
    )
    # Não deve acionar tools de catálogo para receita
    assert not any(t["name"] == "search_products" for t in result.tool_calls)


def test_missing_order_identifier_asks_for_data():
    toolkit = StoreToolkit()
    result = toolkit.check_order_status()
    assert result["status"] == "missing_identifier"
