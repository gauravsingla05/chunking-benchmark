from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    doc_id: str
    path: Path
    text: str


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_via_pdftotext(path: Path) -> str:
    # Uses poppler's pdftotext (already present on your machine).
    # Writes nothing to disk; reads stdout. Suppresses stderr warnings.
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0 or len(result.stdout.strip()) < 50:
        raise RuntimeError(f"pdftotext failed for {path.name} (code {result.returncode})")
    return result.stdout


def load_document(path: Path) -> LoadedDocument:
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _read_pdf_via_pdftotext(path)
    else:
        text = _read_text_file(path)

    doc_id = path.stem
    return LoadedDocument(doc_id=doc_id, path=path, text=text)


def iter_documents(input_dir: Path, *, limit: int | None = None) -> list[LoadedDocument]:
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    allowed_exts = {".pdf", ".txt", ".md"}
    paths = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in allowed_exts and not p.name.startswith(".")
    ]
    if limit is not None:
        paths = paths[: max(0, limit)]

    docs = []
    for p in paths:
        try:
            doc = load_document(p)
            if len(doc.text.strip()) < 100:
                print(f"  SKIP (too short/empty): {p.name}")
                continue
            docs.append(doc)
        except Exception as e:
            print(f"  SKIP (error): {p.name} — {e}")
    return docs

