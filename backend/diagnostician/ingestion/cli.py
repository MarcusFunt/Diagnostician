from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostician.db.session import SessionLocal
from diagnostician.ingestion.parser import LocalCaseIngestor
from diagnostician.services.store import SqlAlchemyGameStore, dump_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local de-identified case source files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("path", help="Source file or directory to ingest.")
    ingest_parser.add_argument("--json", action="store_true", help="Print machine-readable report.")

    args = parser.parse_args()
    if args.command == "ingest":
        reports = ingest(args.path)
        if args.json:
            print(json.dumps([dump_model(report) for report in reports], indent=2))
        else:
            for report in reports:
                status = "playable" if report.playable else "blocked"
                print(f"{report.source_document_id}: {status}")
                for error in report.errors:
                    print(f"  error: {error}")
                for warning in report.warnings:
                    print(f"  warning: {warning}")


def ingest(path: str):
    ingestor = LocalCaseIngestor()
    source_paths = _source_paths(Path(path))
    reports = []
    with SessionLocal() as session:
        store = SqlAlchemyGameStore(session)
        for source_path in source_paths:
            result = ingestor.ingest_path(source_path)
            store.save_source_document(result.source_document)
            if result.truth_case is not None and result.report.accepted:
                store.save_truth_case(result.truth_case, result.embeddings)
            reports.append(result.report)
    return reports


def _source_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.iterdir()
        if item.suffix.lower() in {".json", ".md", ".markdown", ".txt", ".pdf"}
    )


if __name__ == "__main__":
    main()
