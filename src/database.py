"""Camada de banco de dados SQLite, pipeline ETL de tratamento de dados e persistência de conversas."""

from __future__ import annotations

import csv
import json
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, DB_PATH


def normalize_text(text: str | None) -> str:
    """Remove acentos e converte para minúsculas para buscas tolerantes a falhas."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_accents.lower().split())


def _detect_product_quality_notes(
    product_id: int,
    name: str,
    description: str,
    status: str,
    stock_quantity: int,
    specs: dict[str, Any],
) -> list[str]:
    """Detecta inconsistências cadastrais conhecidas nos dados brutos de produtos."""
    notes: list[str] = []
    name_norm = normalize_text(name)
    desc_norm = normalize_text(description)

    # 1. Inconsistência entre status e estoque
    if status == "active" and stock_quantity <= 0:
        notes.append(
            "Status 'active' porém estoque zerado (stock_quantity=0). "
            "Tratado como 'out_of_stock' (indisponível temporariamente)."
        )
    elif status == "discontinued" and stock_quantity > 0:
        notes.append(
            f"Produto marcado como 'discontinued' (descontinuado) apesar de constar "
            f"stock_quantity={stock_quantity}. Conforme política 7.3, não deve ser ofertado como ativo."
        )
    elif status == "coming_soon":
        notes.append(
            "Produto com status 'coming_soon' (lançamento futuro / pré-venda)."
        )

    # 2. Divergência de marcas/modelos entre título e descrição (presente nos IDs 135..144)
    brands = ["music man", "fender", "ibanez", "yamaha", "pearl", "tama", "korg", "roland", "nord", "gibson", "prs"]
    name_brands = [b for b in brands if b in name_norm]
    desc_brands = [b for b in brands if b in desc_norm]
    if name_brands and desc_brands and set(name_brands) != set(desc_brands):
        notes.append(
            f"Divergência cadastral: o nome do produto menciona '{name_brands[0].title()}', "
            f"mas a descrição técnica menciona '{desc_brands[0].title()}'."
        )

    # 3. Divergência de especificações (ex.: 5 cordas na descrição vs 4 cordas no JSON specs, ou 88 vs 61 teclas)
    if "5 cordas" in desc_norm and str(specs.get("strings")) == "4":
        notes.append("Divergência cadastral: descrição cita 5 cordas, mas specs indicam 4 cordas.")
    if "88 teclas" in desc_norm and str(specs.get("keys")) == "61":
        notes.append("Divergência cadastral: descrição cita 88 teclas, mas specs indicam 61 teclas.")

    return notes


class StoreDatabase:
    """Gerenciador SQLite para dados operacionais da loja e histórico de conversas."""

    def __init__(self, db_path: Path = DB_PATH, data_dir: Path = DATA_DIR) -> None:
        self.db_path = db_path
        self.data_dir = data_dir
        self._ensure_initialized()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_initialized(self) -> None:
        """Cria o schema e executa o ETL dos CSVs caso o banco ainda não esteja populado."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            row = cursor.execute("SELECT value FROM metadata WHERE key = 'etl_version'").fetchone()
            if row is None or row["value"] != "2.0":
                self.run_etl(conn)

    def run_etl(self, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
        """Executa o pipeline completo de ingestão, limpeza e normalização dos 6 CSVs."""
        own_conn = False
        if conn is None:
            conn = self.get_connection()
            own_conn = True

        try:
            cur = conn.cursor()

            # Recria tabelas operacionais mantendo as tabelas de histórico de conversa intactas
            cur.executescript(
                """
                DROP TABLE IF EXISTS order_items;
                DROP TABLE IF EXISTS orders;
                DROP TABLE IF EXISTS promotions;
                DROP TABLE IF EXISTS products;
                DROP TABLE IF EXISTS categories;
                DROP TABLE IF EXISTS customers;
                DROP TABLE IF EXISTS data_quality_audit;

                CREATE TABLE categories (
                    category_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_norm TEXT NOT NULL,
                    description TEXT
                );

                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_norm TEXT NOT NULL,
                    category_id INTEGER NOT NULL,
                    category_name TEXT NOT NULL,
                    price_brl REAL NOT NULL,
                    stock_quantity INTEGER NOT NULL,
                    raw_status TEXT NOT NULL,
                    effective_status TEXT NOT NULL,
                    is_available INTEGER NOT NULL,
                    description TEXT,
                    specs_json TEXT,
                    strings_type TEXT,
                    has_electronics INTEGER,
                    color TEXT,
                    created_at TEXT,
                    search_text TEXT NOT NULL,
                    data_quality_notes TEXT,
                    FOREIGN KEY (category_id) REFERENCES categories(category_id)
                );

                CREATE TABLE promotions (
                    promotion_id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL,
                    discount_percent REAL NOT NULL,
                    description TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                );

                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_norm TEXT NOT NULL,
                    phone TEXT,
                    phone_digits TEXT,
                    email TEXT,
                    email_norm TEXT,
                    city TEXT,
                    city_norm TEXT
                );

                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    order_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_brl REAL NOT NULL,
                    items_subtotal_brl REAL NOT NULL,
                    payment_method TEXT NOT NULL,
                    tracking_code TEXT,
                    estimated_delivery TEXT,
                    notes TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                );

                CREATE TABLE order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price_brl REAL NOT NULL,
                    line_total_brl REAL NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id),
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                );

                CREATE TABLE data_quality_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    details TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    customer_context_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                );
                """
            )

            audit_entries: list[tuple[str, str, str, str]] = []

            # 1. Categories
            cat_map: dict[int, str] = {}
            with open(self.data_dir / "desafio_tecnico_ai_eng - categories.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cid = int(row["category_id"])
                    cname = row["name"].strip()
                    cat_map[cid] = cname
                    cur.execute(
                        "INSERT INTO categories VALUES (?, ?, ?, ?)",
                        (cid, cname, normalize_text(cname), row["description"].strip()),
                    )

            # 2. Products
            prod_prices: dict[int, float] = {}
            prod_cats_count: dict[int, int] = {cid: 0 for cid in cat_map}
            with open(self.data_dir / "desafio_tecnico_ai_eng - products.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = int(row["product_id"])
                    price = round(float(row["price_brl"]), 2)
                    prod_prices[pid] = price
                    name = row["name"].strip()
                    cid = int(row["category_id"])
                    prod_cats_count[cid] = prod_cats_count.get(cid, 0) + 1
                    cname = cat_map.get(cid, "Outros")
                    desc = row["description"].strip()
                    stock = int(row["stock_quantity"])
                    raw_status = row["status"].strip().lower()

                    try:
                        specs = json.loads(row["specs"]) if row["specs"] else {}
                    except json.JSONDecodeError:
                        specs = {}

                    # Determina effective_status e disponibilidade real
                    if raw_status == "discontinued":
                        effective_status = "discontinued"
                        is_available = 0
                    elif raw_status == "coming_soon":
                        effective_status = "coming_soon"
                        is_available = 0
                    elif raw_status == "active" and stock <= 0:
                        effective_status = "out_of_stock"
                        is_available = 0
                    else:
                        effective_status = "available"
                        is_available = 1

                    # Extrai atributos estruturados das specs
                    strings_raw = str(specs.get("strings", "")).lower()
                    if "nylon" in strings_raw or "nylon" in normalize_text(name):
                        strings_type = "nylon"
                    elif "steel" in strings_raw or "aco" in normalize_text(name):
                        strings_type = "aco"
                    else:
                        strings_type = strings_raw or None

                    elec_raw = str(specs.get("electronics", "")).lower()
                    has_electronics = (
                        1
                        if elec_raw == "yes"
                        or "eletrico" in normalize_text(name)
                        or "eletroacustico" in normalize_text(desc)
                        else 0
                    )
                    color = specs.get("color")

                    notes = _detect_product_quality_notes(pid, name, desc, raw_status, stock, specs)
                    for n in notes:
                        audit_entries.append(("products", str(pid), "product_inconsistency", n))

                    search_blob = normalize_text(
                        f"{name} {cname} {desc} {json.dumps(specs, ensure_ascii=False)}"
                    )

                    cur.execute(
                        """
                        INSERT INTO products VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            pid,
                            name,
                            normalize_text(name),
                            cid,
                            cname,
                            price,
                            stock,
                            raw_status,
                            effective_status,
                            is_available,
                            desc,
                            json.dumps(specs, ensure_ascii=False),
                            strings_type,
                            has_electronics,
                            color,
                            row["created_at"].strip(),
                            search_blob,
                            " | ".join(notes) if notes else None,
                        ),
                    )

            for cid, count in prod_cats_count.items():
                if count == 0:
                    audit_entries.append(
                        (
                            "categories",
                            str(cid),
                            "empty_category",
                            f"A categoria '{cat_map[cid]}' (ID {cid}) está cadastrada, mas não possui produtos no catálogo.",
                        )
                    )

            # 3. Promotions
            with open(self.data_dir / "desafio_tecnico_ai_eng - promotions.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cur.execute(
                        "INSERT INTO promotions VALUES (?, ?, ?, ?, ?)",
                        (
                            int(row["promotion_id"]),
                            int(row["product_id"]),
                            float(row["discount_percent"]),
                            row["description"].strip(),
                            int(row["is_active"]),
                        ),
                    )

            # 4. Customers
            seen_emails: dict[str, int] = {}
            with open(self.data_dir / "desafio_tecnico_ai_eng - customers.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cust_id = int(row["customer_id"])
                    cname = row["name"].strip()
                    phone = row["phone"].strip()
                    phone_digits = "".join(ch for ch in phone if ch.isdigit())
                    email = row["email"].strip()
                    email_norm = email.lower()
                    city = row["city"].strip()

                    if email_norm in seen_emails:
                        audit_entries.append(
                            (
                                "customers",
                                f"{seen_emails[email_norm]},{cust_id}",
                                "duplicate_email",
                                f"E-mail duplicado '{email_norm}' compartilhado entre customer_id={seen_emails[email_norm]} e customer_id={cust_id}.",
                            )
                        )
                    else:
                        seen_emails[email_norm] = cust_id

                    cur.execute(
                        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            cust_id,
                            cname,
                            normalize_text(cname),
                            phone,
                            phone_digits,
                            email,
                            email_norm,
                            city,
                            normalize_text(city),
                        ),
                    )

            # 5. Order Items (lidos primeiro para calcular subtotal real de cada pedido)
            order_subtotals: dict[int, float] = {}
            order_items_rows: list[tuple[int, int, int, float, float]] = []
            with open(self.data_dir / "desafio_tecnico_ai_eng - order_items.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    oid = int(row["order_id"])
                    pid = int(row["product_id"])
                    qty = int(row["quantity"])
                    uprice = prod_prices.get(pid, 0.0)
                    ltotal = round(qty * uprice, 2)
                    order_subtotals[oid] = round(order_subtotals.get(oid, 0.0) + ltotal, 2)
                    order_items_rows.append((oid, pid, qty, uprice, ltotal))

            # 6. Orders
            with open(self.data_dir / "desafio_tecnico_ai_eng - orders.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    oid = int(row["order_id"])
                    total_brl = round(float(row["total_brl"]), 2)
                    subtotal = order_subtotals.get(oid, total_brl)
                    if abs(total_brl - subtotal) > 0.01:
                        audit_entries.append(
                            (
                                "orders",
                                str(oid),
                                "total_vs_subtotal_diff",
                                f"Pedido #{oid}: total_brl registrado (R$ {total_brl:.2f}) difere da soma de tabela dos itens (R$ {subtotal:.2f}).",
                            )
                        )
                    cur.execute(
                        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            oid,
                            int(row["customer_id"]),
                            row["order_date"].strip(),
                            row["status"].strip().lower(),
                            total_brl,
                            subtotal,
                            row["payment_method"].strip().lower(),
                            row["tracking_code"].strip() or None,
                            row["estimated_delivery"].strip() or None,
                            row["notes"].strip() or None,
                        ),
                    )

            for oid, pid, qty, uprice, ltotal in order_items_rows:
                cur.execute(
                    """
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price_brl, line_total_brl)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (oid, pid, qty, uprice, ltotal),
                )

            for entity, rec_id, issue_type, details in audit_entries:
                cur.execute(
                    "INSERT INTO data_quality_audit (entity, record_id, issue_type, details) VALUES (?, ?, ?, ?)",
                    (entity, rec_id, issue_type, details),
                )

            cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('etl_version', '2.0')"
            )
            cur.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_etl_run', ?)",
                (datetime.now().isoformat(),),
            )
            conn.commit()
            return {"audit_issues_logged": len(audit_entries)}
        finally:
            if own_conn:
                conn.close()

    # ==========================================================================
    # Persistência de Histórico de Conversas
    # ==========================================================================
    def ensure_session(self, session_id: str, title: str | None = None) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT session_id FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO chat_sessions (session_id, title, customer_context_json, created_at, updated_at)
                    VALUES (?, ?, '{}', ?, ?)
                    """,
                    (session_id, title or f"Atendimento {session_id[:8]}", now, now),
                )
            elif title:
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, now, session_id),
                )
            conn.commit()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ensure_session(session_id)
        now = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, tool_calls_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    now,
                ),
            )
            # Atualiza título da sessão com a primeira mensagem do usuário
            if role == "user":
                msg_count = conn.execute(
                    "SELECT COUNT(*) as c FROM chat_messages WHERE session_id = ? AND role = 'user'",
                    (session_id,),
                ).fetchone()["c"]
                if msg_count == 1:
                    short_title = content.strip().replace("\n", " ")[:48]
                    conn.execute(
                        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (short_title, now, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                        (now, session_id),
                    )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
            conn.commit()

    def get_session_messages(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, tool_calls_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            result = []
            for r in rows:
                result.append(
                    {
                        "id": r["id"],
                        "role": r["role"],
                        "content": r["content"],
                        "tool_calls": json.loads(r["tool_calls_json"]) if r["tool_calls_json"] else [],
                        "created_at": r["created_at"],
                    }
                )
            return result

    def update_customer_context(self, session_id: str, context_updates: dict[str, Any]) -> dict[str, Any]:
        self.ensure_session(session_id)
        now = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT customer_context_json FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current = json.loads(row["customer_context_json"]) if row and row["customer_context_json"] else {}
            current.update({k: v for k, v in context_updates.items() if v is not None})
            conn.execute(
                "UPDATE chat_sessions SET customer_context_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(current, ensure_ascii=False), now, session_id),
            )
            conn.commit()
            return current

    def get_customer_context(self, session_id: str) -> dict[str, Any]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT customer_context_json FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row and row["customer_context_json"]:
                return json.loads(row["customer_context_json"])
            return {}

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.customer_context_json, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_session(self, session_id: str) -> None:
        with self.get_connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
