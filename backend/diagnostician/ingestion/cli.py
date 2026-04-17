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
    ingest_parser.add_argument("--limit", type=int, help="Maximum number of source records to ingest.")
    ingest_parser.add_argument("--offset", type=int, default=0, help="Number of source records to skip.")

    seed_parser = subparsers.add_parser("seed-demo")
    seed_parser.add_argument("--path", default="cases/source", help="Demo case directory to ingest.")
    seed_parser.add_argument("--json", action="store_true", help="Print machine-readable report.")

    args = parser.parse_args()
    if args.command == "ingest":
        reports = ingest(args.path, limit=args.limit, offset=args.offset)
        if args.json:
            print(json.dumps([dump_model(report) for report in reports], indent=2))
        else:
            for report in reports:
                status = "playable" if report.playable else "accepted" if report.accepted else "blocked"
                print(f"{report.source_document_id}: {status}")
                for error in report.errors:
                    print(f"  error: {error}")
                for warning in report.warnings:
                    print(f"  warning: {warning}")
    if args.command == "seed-demo":
        reports = ingest(args.path)
        playable = sum(1 for report in reports if report.playable)
        if args.json:
            print(json.dumps([dump_model(report) for report in reports], indent=2))
        else:
            print(f"Seeded {playable} playable demo cases from {args.path}.")
            for report in reports:
                if report.errors or report.warnings:
                    status = "playable" if report.playable else "accepted" if report.accepted else "blocked"
                    print(f"{report.source_document_id}: {status}")
                    for error in report.errors:
                        print(f"  error: {error}")
                    for warning in report.warnings:
                        print(f"  warning: {warning}")


def ingest(path: str, *, limit: int | None = None, offset: int = 0):
    ingestor = LocalCaseIngestor()
    source_paths = _source_paths(Path(path))
    reports = []
    remaining = limit
    with SessionLocal() as session:
        store = SqlAlchemyGameStore(session)
        for index, source_path in enumerate(source_paths):
            path_offset = offset if index == 0 else 0
            for result in ingestor.ingest_path_many(source_path, limit=remaining, offset=path_offset):
                store.save_source_document(result.source_document)
                if result.truth_case is not None and result.report.accepted:
                    store.save_truth_case(result.truth_case, result.embeddings)
                reports.append(result.report)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
            if remaining is not None and remaining <= 0:
                break
    return reports


def _source_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.iterdir()
        if item.suffix.lower() in {".json", ".md", ".markdown", ".txt", ".pdf", ".parquet"}
    )


if __name__ == "__main__":
    main()
