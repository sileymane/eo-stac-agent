"""CLI entry point: python -m eo_stac_agent "Sentinel-2 over Toulouse in July 2025, <10% cloud" """
import argparse
import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    if os.getenv("USE_SYSTEM_CERTS", "true").lower() == "true":
        # Trust the Windows/macOS certificate store (needed behind TLS-inspecting corporate proxies).
        import truststore

        truststore.inject_into_ssl()

    from .graph import build_graph

    parser = argparse.ArgumentParser(description="Natural-language STAC search agent")
    parser.add_argument("query", nargs="?", help="request; omit for interactive mode")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    graph = build_graph()
    if args.graph:
        print(graph.get_graph().draw_mermaid())
        return

    if args.query:
        run(graph, args.query)
        return

    print("EO data search - type a request, or 'quit'.")
    while (query := input("\n> ").strip()).lower() not in {"quit", "exit", "q"}:
        if query:
            run(graph, query)


def run(graph, query: str) -> None:
    answer = ""
    # stream_mode="updates" yields {node_name: state_update} after each node: handy for seeing the loop.
    for update in graph.stream({"query": query}, stream_mode="updates"):
        for node, change in update.items():
            print(f"  [{node}] {_describe(node, change)}")
            answer = change.get("answer", answer)
    print("\n" + answer)


def _describe(node: str, change: dict) -> str:
    if change.get("error"):
        return f"error: {change['error']}"
    if node == "parse_request":
        return change["request"].model_dump_json(exclude_none=True)
    if node == "geocode":
        return f"bbox={[round(v, 3) for v in change['bbox']]}"
    if node == "search":
        return f"attempt {change['attempts']}: {len(change['items'])} result(s)"
    if node == "relax":
        return change["relaxations"][-1]
    return "done"


if __name__ == "__main__":
    main()
