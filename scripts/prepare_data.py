"""Localiza ou baixa, mediante solicitação explícita, o dataset Housing in London."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

DATASET_HANDLE = "justinas/housing-in-london"
REQUIRED_FILES = (
    "housing_in_london_monthly_variables.csv",
    "housing_in_london_yearly_variables.csv",
)


def _complete(folder: Path) -> bool:
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in REQUIRED_FILES)


def ensure_data(directory: str | Path | None = None, *, allow_download: bool = False) -> Path:
    """Retorna diretório com CSVs. Rede somente com allow_download=True.

    Em modo sem diretório explícito, aceita a pasta data/, o cwd (legado)
    e /content (Colab legado). Nunca distribui o dataset no repositório.
    """
    explicit = directory is not None or bool(os.getenv("LONDON_DATA_DIR"))
    target = Path(directory or os.getenv("LONDON_DATA_DIR", "data")).expanduser().resolve()
    candidates = [target] if explicit else [target, Path.cwd().resolve(), Path("/content")]
    for candidate in candidates:
        if _complete(candidate):
            return candidate

    if not allow_download:
        raise FileNotFoundError(
            "Os CSVs de Housing in London não foram encontrados. "
            "Baixe-os da fonte Kaggle para data/ ou execute "
            "`python scripts/prepare_data.py --download` para autorizar "
            "o download via API oficial."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Instale as dependências: pip install -r requirements.txt") from exc

    target.mkdir(parents=True, exist_ok=True)
    try:
        obtained = Path(kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(target)))
    except Exception as exc:
        raise RuntimeError(
            "Falha ao obter os dados pela API oficial do Kaggle. "
            "Confira a disponibilidade, os termos de uso e, se exigido, "
            "a autenticação do Kaggle; alternativamente, copie os CSVs manualmente."
        ) from exc

    # Tolera tanto saída plana quanto arquivos extraídos em subpasta.
    for name in REQUIRED_FILES:
        destination = target / name
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        matches = [p for p in obtained.rglob(name) if p.is_file() and p.stat().st_size > 0]
        if matches and matches[0].resolve() != destination.resolve():
            shutil.copy2(matches[0], destination)

    if not _complete(target):
        missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
        raise FileNotFoundError(f"Download incompleto; faltam arquivos: {missing}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o dataset Housing in London sem redistribuí-lo")
    parser.add_argument("--download", action="store_true", help="Autoriza download pela API oficial do Kaggle")
    parser.add_argument("--dir", default=None, help="Pasta dos dados (padrão: data/)")
    args = parser.parse_args()
    data_dir = ensure_data(args.dir, allow_download=args.download)
    print(f"CSVs disponíveis em: {data_dir}")
    for name in REQUIRED_FILES:
        print(f"- {name}: {(data_dir / name).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
