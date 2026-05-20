# main.py
"""
CLI entry point.
Shared pipeline state across commands via persistent SQLite + Qdrant.
"""

import argparse
import logging
import sys

from pipeline import AgenticRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agentic_rag.log"),
    ],
)
logger = logging.getLogger(__name__)


def get_pipeline(index_dir: str = None) -> AgenticRAGPipeline:
    return AgenticRAGPipeline(index_dir=index_dir)


def cmd_ingest(args):
    pipeline = get_pipeline(args.index_dir)
    stats = pipeline.ingest(args.files)

    print("\n✅ Ingestion Complete!")
    print(f"   Files processed: {stats['files_processed']}")
    print(f"   Files skipped: {stats.get('files_skipped', 0)}")
    print(f"   Pages: {stats['total_pages']}")
    print(f"   Parent chunks: {stats['parent_chunks']}")
    print(f"   Child chunks: {stats['child_chunks']}")

    status = pipeline.status()
    print(f"\n   Index status:")
    print(f"   Qdrant points: {status['qdrant_points']}")
    print(f"   Parent chunks: {status['parent_chunks']}")
    print(f"   BM25 documents: {status['bm25_documents']}")

    pipeline.close()


def cmd_query(args):
    pipeline = get_pipeline(args.index_dir)

    status = pipeline.status()
    if status["qdrant_points"] == 0:
        print("⚠️  Warning: Qdrant has no vectors. Did you run `ingest` first?")
    if status["parent_chunks"] == 0:
        print("⚠️  Warning: No parent chunks loaded. Parent expansion will fail.")
    if status["bm25_documents"] == 0:
        print("⚠️  Warning: BM25 index is empty. Retrieval is vector-only.")

    result = pipeline.query(args.question)

    print("\n" + "=" * 80)
    print("📝 ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "-" * 80)
    print("📊 EVALUATION")
    print("-" * 80)
    print(f"  Successful: {result['is_successful']}")
    print(f"  Retrieval: {result['retrieval_decision']}")
    print(f"  Relevance: {result['relevance_grade']}")
    print(f"  Support: {result['support_grade']}")
    print(f"  Usefulness: {result['usefulness_score']}/5")
    print(f"  Retries: {result['num_retries']}")

    if result["transformations"]:
        print("  Query Transformations:")
        for t in result["transformations"]:
            print(f"    - {t}")

    if result["documents"]:
        print(f"\n📄 SOURCES ({len(result['documents'])} documents)")
        for i, doc in enumerate(result["documents"][:5], 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  [{i}] {source} (page {page})")

    pipeline.close()


def cmd_interactive(args):
    pipeline = get_pipeline(args.index_dir)

    status = pipeline.status()
    print(f"\n🤖 Agentic RAG — Interactive Mode")
    print(
        f"   Qdrant: {status['qdrant_points']} points | "
        f"Parents: {status['parent_chunks']} | "
        f"BM25: {status['bm25_documents']} docs"
    )
    print(
        "Type 'quit' to exit, 'reset' to clear data, 'status' for info\n"
    )

    while True:
        try:
            question = input("❓ Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            pipeline.reset()
            print("🗑️  Pipeline reset.\n")
            continue
        if question.lower() == "status":
            s = pipeline.status()
            print(
                f"   Qdrant: {s['qdrant_points']} | "
                f"Parents: {s['parent_chunks']} | "
                f"BM25: {s['bm25_documents']}\n"
            )
            continue

        result = pipeline.query(question)
        print(f"\n💡 {result['answer']}")
        print(
            f"   [✓ Success: {result['is_successful']} | "
            f"Support: {result['support_grade']} | "
            f"Usefulness: {result['usefulness_score']}/5 | "
            f"Retries: {result['num_retries']}]"
        )
        print()

    pipeline.close()


def cmd_status(args):
    pipeline = get_pipeline(args.index_dir)
    status = pipeline.status()

    print("\n📊 Index Status")
    print(f"   Index dir:       {status['index_dir']}")
    print(f"   Qdrant collection: {status['collection_name']}")
    print(f"   Qdrant points:   {status['qdrant_points']}")
    print(f"   Parent chunks:   {status['parent_chunks']}")
    print(f"   BM25 documents:  {status['bm25_documents']}")

    if status["qdrant_points"] == 0:
        print(
            "\n   ⚠️  Index is empty. Run `python main.py ingest --files ...` first."
        )
    elif status["parent_chunks"] == 0 or status["bm25_documents"] == 0:
        print(
            "\n   ⚠️  Index is partial. Some stores are empty (persistence issue?)."
        )

    pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory for persistent index data (default: ./index_data)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--files", nargs="+", required=True)

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", type=str)

    subparsers.add_parser("interactive", help="Interactive query mode")
    subparsers.add_parser("status", help="Show index status")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
