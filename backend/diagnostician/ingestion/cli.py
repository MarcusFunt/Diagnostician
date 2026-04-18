from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from sqlalchemy import select

from diagnostician.core.schemas import CaseFact
from diagnostician.db.models import CaseFactEmbeddingRow, CaseFactRow, IngestionRunRow
from diagnostician.db.session import SessionLocal
from diagnostician.ingestion.parser import LocalCaseIngestor
from diagnostician.llm.ollama_client import OllamaClient
from diagnostician.services.store import SqlAlchemyGameStore, dump_model


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MULTICARE_DIR = ROOT / "data" / "multicare-cases"
DEFAULT_MULTICARE_FILE = ROOT / "cases.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local de-identified case source files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("path", help="Source file or directory to ingest.")
    ingest_parser.add_argument("--json", "--json-report", action="store_true", help="Print machine-readable report.")
    ingest_parser.add_argument("--limit", type=int, help="Maximum number of source records to ingest.")
    ingest_parser.add_argument("--offset", type=int, default=0, help="Number of source records to skip.")
    ingest_parser.add_argument("--batch-size", type=int, default=128, help="Parquet rows to read per batch.")
    ingest_parser.add_argument("--skip-embeddings", action="store_true", help="Persist cases without embedding facts.")

    seed_parser = subparsers.add_parser("seed-demo")
    seed_parser.add_argument("--path", default="cases/source", help="Demo case directory to ingest.")
    seed_parser.add_argument("--json", "--json-report", action="store_true", help="Print machine-readable report.")

    pull_parser = subparsers.add_parser("pull-multicare")
    pull_parser.add_argument("--local-dir", default=str(DEFAULT_MULTICARE_DIR), help="Download directory.")
    pull_parser.add_argument("--json", "--json-report", action="store_true", help="Print machine-readable report.")

    multicare_parser = subparsers.add_parser("ingest-multicare")
    multicare_parser.add_argument("path", nargs="?", help="Parquet source path. Defaults to local-first discovery.")
    multicare_parser.add_argument("--json", "--json-report", action="store_true", help="Print machine-readable report.")
    multicare_parser.add_argument("--limit", type=int, help="Maximum number of case records to ingest.")
    multicare_parser.add_argument("--offset", type=int, default=0, help="Number of case records to skip.")
    multicare_parser.add_argument("--batch-size", type=int, default=500, help="Parquet rows to read per batch.")
    multicare_parser.add_argument("--resume", action="store_true", help="Resume from the last saved offset for this source hash.")
    multicare_parser.add_argument("--skip-embeddings", action="store_true", help="Persist playable cases before embedding backfill.")
    multicare_parser.add_argument("--llm-extract", action="store_true", help="Use structured LLM extraction only when deterministic parsing fails.")

    ingest_all_parser = subparsers.add_parser("ingest-all")
    ingest_all_parser.add_argument("--demo-path", default="cases/source", help="Demo case directory to seed first.")
    ingest_all_parser.add_argument("--multicare-path", help="Optional MultiCaRe parquet path.")
    ingest_all_parser.add_argument("--json", "--json-report", action="store_true", help="Print machine-readable report.")
    ingest_all_parser.add_argument("--limit", type=int, help="Maximum number of MultiCaRe case records to ingest.")
    ingest_all_parser.add_argument("--offset", type=int, default=0, help="Number of MultiCaRe case records to skip.")
    ingest_all_parser.add_argument("--batch-size", type=int, default=500, help="Parquet rows to read per batch.")
    ingest_all_parser.add_argument("--resume", action="store_true", help="Resume MultiCaRe ingestion from the saved offset.")
    ingest_all_parser.add_argument("--skip-embeddings", action="store_true", help="Persist playable cases before embedding backfill.")
    ingest_all_parser.add_argument("--llm-extract", action="store_true", help="Use structured LLM extraction only when deterministic parsing fails.")

    backfill_parser = subparsers.add_parser("backfill-embeddings")
    backfill_parser.add_argument("--limit", type=int, help="Maximum number of facts to embed.")
    backfill_parser.add_argument("--batch-size", type=int, default=250, help="Facts to embed per database batch.")
    backfill_parser.add_argument("--json", "--json-report", action="store_true", help="Print machine-readable report.")

    args = parser.parse_args()
    if args.command == "ingest":
        reports = ingest(
            args.path,
            limit=args.limit,
            offset=args.offset,
            batch_size=args.batch_size,
            skip_embeddings=args.skip_embeddings,
        )
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
    if args.command == "pull-multicare":
        path = pull_multicare(Path(args.local_dir))
        payload = {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"MultiCaRe parquet ready at {path}")
    if args.command == "ingest-multicare":
        summary = ingest_multicare(
            args.path,
            limit=args.limit,
            offset=args.offset,
            batch_size=args.batch_size,
            resume=args.resume,
            skip_embeddings=args.skip_embeddings,
            llm_extract=args.llm_extract,
        )
        print(json.dumps(summary, indent=2) if args.json else _format_multicare_summary(summary))
    if args.command == "ingest-all":
        demo_reports = ingest(args.demo_path)
        multicare_summary = ingest_multicare(
            args.multicare_path,
            limit=args.limit,
            offset=args.offset,
            batch_size=args.batch_size,
            resume=args.resume,
            skip_embeddings=args.skip_embeddings,
            llm_extract=args.llm_extract,
        )
        payload = {
            "demo": {
                "processed": len(demo_reports),
                "playable": sum(1 for report in demo_reports if report.playable),
            },
            "multicare": multicare_summary,
        }
        print(json.dumps(payload, indent=2) if args.json else _format_multicare_summary(multicare_summary))
    if args.command == "backfill-embeddings":
        summary = backfill_embeddings(limit=args.limit, batch_size=args.batch_size)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"Embedded {summary['embedded']} facts; "
                f"{summary['fallback_embeddings']} used deterministic fallback vectors."
            )


def ingest(
    path: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = 128,
    skip_embeddings: bool = False,
):
    ingestor = LocalCaseIngestor(generate_embeddings=not skip_embeddings)
    source_paths = _source_paths(Path(path))
    reports = []
    remaining = limit
    with SessionLocal() as session:
        store = SqlAlchemyGameStore(session)
        for index, source_path in enumerate(source_paths):
            path_offset = offset if index == 0 else 0
            for result in ingestor.ingest_path_many(
                source_path,
                limit=remaining,
                offset=path_offset,
                batch_size=batch_size,
            ):
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


def resolve_multicare_source(*, download_if_missing: bool = False, local_dir: Path = DEFAULT_MULTICARE_DIR) -> Path:
    if DEFAULT_MULTICARE_FILE.exists():
        return DEFAULT_MULTICARE_FILE
    downloaded = local_dir / "cases.parquet"
    if downloaded.exists():
        return downloaded
    if download_if_missing:
        return pull_multicare(local_dir)
    raise FileNotFoundError(
        f"No MultiCaRe parquet found at {DEFAULT_MULTICARE_FILE} or {downloaded}."
    )


def pull_multicare(local_dir: Path = DEFAULT_MULTICARE_DIR) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "download",
        "OpenMed/multicare-cases",
        "--repo-type",
        "dataset",
        "--include",
        "*.parquet",
        "--include",
        "README.md",
        "--local-dir",
        str(local_dir),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("The Hugging Face `hf` CLI is required to download MultiCaRe.") from exc
    return local_dir / "cases.parquet"


def ingest_multicare(
    path: str | None = None,
    *,
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = 500,
    resume: bool = False,
    skip_embeddings: bool = False,
    llm_extract: bool = False,
) -> dict:
    source_path = Path(path) if path else resolve_multicare_source(download_if_missing=True)
    source_hash = _sha256_file(source_path)
    ingestor = LocalCaseIngestor(
        generate_embeddings=not skip_embeddings,
        use_llm_extraction=llm_extract,
    )
    processed = 0
    accepted = 0
    playable = 0
    skipped = 0
    errors = 0
    error_samples: list[dict] = []

    with SessionLocal() as session:
        store = SqlAlchemyGameStore(session)
        progress = _start_or_resume_ingestion_run(
            session,
            source_path=source_path,
            source_hash=source_hash,
            offset=offset,
            resume=resume,
            payload={
                "limit": limit,
                "batch_size": batch_size,
                "skip_embeddings": skip_embeddings,
                "llm_extract": llm_extract,
            },
        )
        base_accepted = progress.accepted_count
        base_playable = progress.playable_count
        base_skipped = progress.skipped_count
        base_errors = progress.error_count
        start_offset = progress.current_offset if resume else offset
        try:
            for result in ingestor.ingest_path_many(
                source_path,
                limit=limit,
                offset=start_offset,
                batch_size=batch_size,
            ):
                processed += 1
                progress.current_offset = start_offset + processed
                if result.report.accepted and result.truth_case is not None:
                    store.save_source_document(result.source_document)
                    store.save_truth_case(result.truth_case, result.embeddings)
                    accepted += 1
                    if result.report.playable:
                        playable += 1
                else:
                    skipped += 1
                    if result.report.errors:
                        errors += 1
                        if len(error_samples) < 20:
                            error_samples.append(
                                {
                                    "source_document_id": str(result.report.source_document_id),
                                    "errors": result.report.errors,
                                }
                            )
                if processed % batch_size == 0:
                    _update_progress(
                        session,
                        progress,
                        accepted=base_accepted + accepted,
                        playable=base_playable + playable,
                        skipped=base_skipped + skipped,
                        errors=base_errors + errors,
                        status="running",
                    )
            _update_progress(
                session,
                progress,
                accepted=base_accepted + accepted,
                playable=base_playable + playable,
                skipped=base_skipped + skipped,
                errors=base_errors + errors,
                status="completed",
            )
        except Exception as exc:
            progress.last_error = str(exc)
            _update_progress(
                session,
                progress,
                accepted=base_accepted + accepted,
                playable=base_playable + playable,
                skipped=base_skipped + skipped,
                errors=base_errors + errors + 1,
                status="failed",
            )
            raise

    return {
        "source_path": str(source_path),
        "source_hash": source_hash,
        "start_offset": start_offset,
        "processed": processed,
        "accepted": accepted,
        "playable": playable,
        "skipped": skipped,
        "errors": errors,
        "error_samples": error_samples,
        "skip_embeddings": skip_embeddings,
    }


def backfill_embeddings(*, limit: int | None = None, batch_size: int = 250) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    client = OllamaClient()
    embedded = 0
    fallback_embeddings = 0
    with SessionLocal() as session:
        while limit is None or embedded < limit:
            current_limit = batch_size if limit is None else min(batch_size, limit - embedded)
            if current_limit <= 0:
                break
            rows = session.scalars(
                select(CaseFactRow)
                .outerjoin(CaseFactEmbeddingRow, CaseFactEmbeddingRow.fact_id == CaseFactRow.id)
                .where(CaseFactEmbeddingRow.id.is_(None))
                .limit(current_limit)
            ).all()
            if not rows:
                break
            for row in rows:
                fact = CaseFact.model_validate(row.payload)
                result = client.embed(fact.search_text)
                session.add(
                    CaseFactEmbeddingRow(
                        fact_id=fact.id,
                        embedding_model=result.model,
                        embedding=result.vector,
                    )
                )
                embedded += 1
                if result.fallback_used:
                    fallback_embeddings += 1
            session.commit()
    return {"embedded": embedded, "fallback_embeddings": fallback_embeddings}


def _source_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.iterdir()
        if item.suffix.lower() in {".json", ".md", ".markdown", ".txt", ".pdf", ".parquet"}
    )


def _start_or_resume_ingestion_run(
    session,
    *,
    source_path: Path,
    source_hash: str,
    offset: int,
    resume: bool,
    payload: dict,
) -> IngestionRunRow:
    row = None
    if resume:
        row = session.scalars(
            select(IngestionRunRow)
            .where(IngestionRunRow.source_hash == source_hash)
            .order_by(IngestionRunRow.created_at.desc())
            .limit(1)
        ).first()
    if row is None:
        row = IngestionRunRow(
            source_path=str(source_path),
            source_hash=source_hash,
            current_offset=offset,
            status="running",
            payload=payload,
        )
        session.add(row)
    else:
        row.status = "running"
        row.last_error = None
        row.payload = {**(row.payload or {}), **payload, "resumed": True}
    session.commit()
    return row


def _update_progress(
    session,
    row: IngestionRunRow,
    *,
    accepted: int,
    playable: int,
    skipped: int,
    errors: int,
    status: str,
) -> None:
    row.accepted_count = accepted
    row.playable_count = playable
    row.skipped_count = skipped
    row.error_count = errors
    row.status = status
    session.commit()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_multicare_summary(summary: dict) -> str:
    return (
        f"MultiCaRe ingestion processed {summary['processed']} records from {summary['source_path']}. "
        f"Accepted {summary['accepted']} playable {summary['playable']}; "
        f"skipped {summary['skipped']} with {summary['errors']} errors."
    )


if __name__ == "__main__":
    main()
