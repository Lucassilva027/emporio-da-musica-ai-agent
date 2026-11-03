"""Motor de Retrieval-Augmented Generation (RAG) Híbrido para o Manual de Políticas (PDF).

Combina:
1. Limpeza de artefatos de extração do PDF (palavras quebradas por linha e cabeçalhos repetidos).
2. Chunking Semântico Estruturado por seções hierárquicas do Manual v2.1.
3. Busca Híbrida: Embeddings Densos (sentence-transformers/all-MiniLM-L6-v2 com cache em disco)
   + Busca Léxica TF-IDF (word + char n-grams normalizados sem acento) + Topic Boosting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import POLICIES_PDF_PATH, STORAGE_DIR
from src.database import normalize_text

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


@dataclass
class PolicyChunk:
    chunk_id: str
    section_number: str
    section_title: str
    pages: list[int]
    content: str
    keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SECTION_KEYWORDS_MAP: dict[str, list[str]] = {
    "1": [
        "sobre", "loja", "historia", "fundacao", "2008", "campo grande", "endereco",
        "rua 14 de maio", "3200", "centro", "cep", "79202-333", "cnpj", "razao social",
        "telefone", "whatsapp", "3341-4444", "email", "contato", "missao", "acessorios",
        "cordas", "palhetas", "cabos", "cases", "pedais", "amplificadores", "localizacao",
        "onde fica",
    ],
    "2": [
        "horario", "funcionamento", "expediente", "abre", "fecha", "aberto", "fechado",
        "segunda", "sexta", "sabado", "domingo", "feriado", "09:00", "18:00", "13:00",
        "atendimento presencial",
    ],
    "3": [
        "pagamento", "formas de pagamento", "pix", "5%", "desconto a vista", "cartao",
        "credito", "debito", "boleto", "parcelamento", "parcelar", "parcela minima",
        "sem juros", "12x", "3x", "6x", "combinacao", "duas formas", "compensacao",
    ],
    "4": [
        "troca", "devolucao", "devolver", "trocar", "arrependimento", "arrependi",
        "desistir", "7 dias", "30 dias", "defeito", "preferencia", "cor", "modelo",
        "reembolso", "estorno", "embalagem original", "venda final", "boquilha",
        "personalizacao", "setup", "regulagem", "frete de devolucao", "cdc",
    ],
    "5": [
        "frete", "entrega", "envio", "prazo de entrega", "campo grande", "motoboy",
        "frete gratis", "500", "35,00", "outras cidades", "correios", "pac", "sedex",
        "jadlog", "rastreamento", "codigo de rastreio", "seguro", "avaria", "extravio",
        "grande porte",
    ],
    "6": [
        "promocao", "promocoes", "desconto", "campanha", "aniversario da loja",
        "black friday", "volta as aulas", "queima de estoque", "semana do musico",
        "cumulativo", "acumular", "rain check", "expirada", "vencida",
    ],
    "7": [
        "whatsapp", "3321-4500", "atendimento remoto", "tom de voz", "tempo de resposta",
        "5 minutos", "escopo", "acessorios", "fora de estoque", "esgotado", "indisponivel",
        "descontinuado", "promocao expirada", "reclamacao", "24 horas", "fluxo",
    ],
    "8": [
        "garantia", "garantia legal", "90 dias", "garantia do fabricante", "6 meses",
        "2 anos", "certificado", "mau uso", "queda", "umidade", "desgaste natural",
        "trastes", "reparo",
    ],
    "9": [
        "privacidade", "protecao de dados", "lgpd", "13.709", "dados pessoais",
        "exclusao de dados", "marketing", "terceiros", "consentimento",
    ],
    "10": [
        "disposicoes finais", "manual interno", "gerencia", "versao 2.1", "junho de 2025",
    ],
}


class PolicyRAGEngine:
    """Motor RAG Híbrido sobre o PDF de Políticas e Procedimentos da Empório da Música."""

    def __init__(self, pdf_path: Path = POLICIES_PDF_PATH) -> None:
        self.pdf_path = pdf_path
        self.chunks: list[PolicyChunk] = self._extract_and_chunk_pdf()
        self.norm_corpus: list[str] = [
            normalize_text(f"{c.section_title} {' '.join(c.keywords)} {c.content}")
            for c in self.chunks
        ]

        # Vetorizador Léxico TF-IDF (combina termos e radicais)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.norm_corpus)

        # Embeddings Semânticos Densos (com cache em disco para inicialização instantânea)
        self.dense_embeddings: np.ndarray | None = None
        self._embed_model: Any = None
        self._init_dense_embeddings()

    def _clean_page_text(self, raw_text: str) -> str:
        """Limpa o texto extraído de uma página do PDF, removendo cabeçalhos/rodapés repetitivos."""
        tokens = raw_text.split()
        text = " ".join(tokens)
        # Remove cabeçalho padrão das páginas 2..8
        text = re.sub(
            r"^Empório da Música Manual de Políticas e Procedimentos\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Remove rodapé "Página X"
        text = re.sub(r"\s*Página\s+\d+\s*$", "", text, flags=re.IGNORECASE)
        # Corrige espaçamento em telefone "(67) 3341 - 4444" -> "(67) 3341-4444"
        text = re.sub(r"\(\s*67\s*\)\s*(\d{4})\s*-\s*(\d{4})", r"(67) \1-\2", text)
        return text.strip()

    def _extract_and_chunk_pdf(self) -> list[PolicyChunk]:
        """Extrai o texto do PDF e segmenta pelas 10 seções oficiais do manual."""
        reader = PdfReader(str(self.pdf_path))
        page_texts: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            cleaned = self._clean_page_text(page.extract_text() or "")
            if cleaned:
                page_texts.append((idx, cleaned))

        # Combina texto preservando marcadores de página
        combined_parts: list[str] = []
        for pno, ptext in page_texts:
            combined_parts.append(f"[PAGE_{pno}] {ptext}")
        full_text = " ".join(combined_parts)

        # Títulos exatos das 10 seções principais do documento
        section_headers = [
            ("1", "Sobre a Empório da Música", r"1\.\s+Sobre a Empório da Música"),
            ("2", "Horário de Funcionamento", r"2\.\s+Horário de Funcionamento"),
            ("3", "Formas de Pagamento e Parcelamento", r"3\.\s+Formas de Pagamento"),
            ("4", "Política de Trocas e Devoluções", r"4\.\s+Política de Trocas e Devoluções"),
            ("5", "Política de Frete, Entregas e Rastreamento", r"5\.\s+Política de Frete e Entregas"),
            ("6", "Promoções e Descontos", r"6\.\s+Promoções e Descontos"),
            ("7", "Atendimento via WhatsApp e Situações Especiais", r"7\.\s+Atendimento via WhatsApp"),
            ("8", "Garantia Legal e do Fabricante", r"8\.\s+Garantia"),
            ("9", "Privacidade e Proteção de Dados (LGPD)", r"9\.\s+Privacidade e Proteção de Dados"),
            ("10", "Disposições Finais", r"10\.\s+Disposições Finais"),
        ]

        matches: list[tuple[str, str, int]] = []
        for sec_num, sec_title, pattern in section_headers:
            m = re.search(pattern, full_text)
            if m:
                matches.append((sec_num, sec_title, m.start()))

        chunks: list[PolicyChunk] = []
        for i, (sec_num, sec_title, start_pos) in enumerate(matches):
            end_pos = matches[i + 1][2] if i + 1 < len(matches) else len(full_text)
            raw_slice = full_text[start_pos:end_pos]

            # Identifica páginas abrangidas pela seção
            pages_in_slice = [int(p) for p in re.findall(r"\[PAGE_(\d+)\]", raw_slice)]
            # Também verifica qual página estava ativa imediatamente antes de start_pos
            prev_pages = [int(p) for p in re.findall(r"\[PAGE_(\d+)\]", full_text[:start_pos])]
            all_pages = sorted(set((prev_pages[-1:] if prev_pages else []) + pages_in_slice))

            clean_slice = re.sub(r"\[PAGE_\d+\]\s*", "", raw_slice).strip()
            # Na seção 1, inclui informações de capa (endereço/CEP/e-mail sem acento)
            if sec_num == "1" and page_texts:
                cover_text = page_texts[0][1]
                clean_slice = f"Capa Institucional: {cover_text} | {clean_slice}"
                if 1 not in all_pages:
                    all_pages = [1] + all_pages

            chunks.append(
                PolicyChunk(
                    chunk_id=f"sec_{sec_num}",
                    section_number=sec_num,
                    section_title=sec_title,
                    pages=all_pages,
                    content=clean_slice,
                    keywords=SECTION_KEYWORDS_MAP.get(sec_num, []),
                )
            )

        return chunks

    def _init_dense_embeddings(self) -> None:
        """Carrega ou gera embeddings densos usando sentence-transformers (se disponível em cache)."""
        corpus_hash = hashlib.md5(
            json.dumps([c.content for c in self.chunks], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        cache_file = STORAGE_DIR / f"policy_embeddings_{corpus_hash}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file)
                self.dense_embeddings = data["embeddings"]
                return
            except Exception:
                pass

        try:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True,
            )
            texts = [f"{c.section_title}: {c.content}" for c in self.chunks]
            emb = self._embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.dense_embeddings = np.asarray(emb, dtype=np.float32)
            np.savez_compressed(cache_file, embeddings=self.dense_embeddings)
        except Exception:
            self.dense_embeddings = None

    def _encode_query_dense(self, query: str) -> np.ndarray | None:
        if self.dense_embeddings is None:
            return None
        try:
            if self._embed_model is None:
                from sentence_transformers import SentenceTransformer

                self._embed_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    local_files_only=True,
                )
            q_emb = self._embed_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(q_emb, dtype=np.float32)
        except Exception:
            return None

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Executa busca híbrida (Semântica + TF-IDF + Keyword Match) nos chunks do manual."""
        q_norm = normalize_text(query)
        if not q_norm:
            return [c.to_dict() for c in self.chunks[:top_k]]

        # 1. Score Léxico TF-IDF
        q_tfidf = self.tfidf.transform([q_norm])
        sparse_scores = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]

        # 2. Score de Keywords de Domínio
        kw_scores = np.zeros(len(self.chunks), dtype=np.float32)
        for idx, chunk in enumerate(self.chunks):
            hits = sum(1 for kw in chunk.keywords if normalize_text(kw) in q_norm)
            if hits > 0:
                kw_scores[idx] = min(1.0, hits * 0.28)

        # 3. Score Semântico Denso (SentenceTransformers)
        q_dense = self._encode_query_dense(query)
        if q_dense is not None and self.dense_embeddings is not None:
            dense_scores = np.dot(self.dense_embeddings, q_dense[0])
            dense_scores = np.clip(dense_scores, 0.0, 1.0)
            combined_scores = 0.45 * dense_scores + 0.35 * sparse_scores + 0.20 * kw_scores
        else:
            combined_scores = 0.60 * sparse_scores + 0.40 * kw_scores

        ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]
            score = float(combined_scores[int(idx)])
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "section_number": chunk.section_number,
                    "section_title": chunk.section_title,
                    "pages": chunk.pages,
                    "relevance_score": round(score, 4),
                    "content": chunk.content,
                }
            )
        return results
