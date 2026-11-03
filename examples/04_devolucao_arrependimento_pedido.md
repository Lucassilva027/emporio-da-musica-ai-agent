# Cenário não trivial: cruza política de arrependimento (7 dias) com dados reais do pedido #1.

**Provedor:** `local` / `local-hybrid-react-v1`

## Cliente

> Me arrependi da minha compra, posso devolver meu pedido #1?

## Tom (Assistente Empório da Música)

**Direito de Arrependimento (compras online — Seção 4.1):**
• Até **7 dias corridos** após o recebimento, sem precisar justificar.
• Produto na embalagem original, sem sinais de uso, com acessórios e manuais.
• Frete de devolução por conta da loja.
• Reembolso na mesma forma de pagamento em até **10 dias úteis**.


**Aplicando ao seu Pedido #1** (Entregue, cliente Pedro Henrique Oliveira):
Pedido comprado em 2025-10-15 (159 dias atrás) e entregue em 2025-10-25 (149 dias atrás, considerando a data de referência 2026-03-23). - Direito de Arrependimento / Troca por Preferência (7 dias corridos do recebimento): FORA DO PRAZO (149 dias corridos desde a entrega; limite é 7 dias). - Troca direta na loja por defeito de fabricação (30 dias corridos da compra): FORA DO PRAZO (159 dias desde a compra; limite na loja é 30 dias, acionar garantia do fabricante com intermediação da loja). - Garantia Legal CDC (90 dias do recebimento): FORA DO PRAZO DE 90 DIAS (verificar garantia contratual do fabricante de 6 meses a 2 anos).
• Arrependimento 7 dias: ❌ fora do prazo
• Troca por defeito 30 dias (loja): ❌ fora do prazo / acionar fabricante
• Garantia legal 90 dias: ❌ fora do prazo


_Data de referência do sistema: 2026-03-23._

Quer que eu consulte um pedido específico?

## Ferramentas acionadas

- `search_store_policies` — args: `{'query': 'Me arrependi da minha compra, posso devolver meu pedido #1?', 'top_k': 3}`
- `check_order_status` — args: `{'order_id': 1}`
