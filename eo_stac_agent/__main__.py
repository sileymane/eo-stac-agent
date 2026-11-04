"""CLI entry point: python -m eo_stac_agent "Sentinel-2 over Toulouse in July 2025, <10% cloud" """
import argparse
import os
import uuid

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    if os.getenv("USE_SYSTEM_CERTS", "true").lower() == "true":
        # Trust the Windows/macOS certificate store (needed behind TLS-inspecting corporate proxies).
        import truststore

        truststore.inject_into_ssl()

    from .graph import build_graph, make_checkpointer

    parser = argparse.ArgumentParser(description="Natural-language STAC search agent")
    parser.add_argument("query", nargs="?", help="request; omit for interactive mode")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    args = parser.parse_args()

    # The checkpointer keeps each thread's state between turns, which is what makes follow-ups work.
    graph = build_graph(checkpointer=make_checkpointer())
    if args.graph:
        print(graph.get_graph().draw_mermaid())
        return

    thread_id = new_thread()
    if args.query:
        run(graph, args.query, thread_id)
        return

    print("EO data search - type a request, 'new' to start over, or 'quit'.")
    print("Follow-ups work: 'same thing but for August', 'what about Landsat?'")
    while (query := input("\n> ").strip()).lower() not in {"quit", "exit", "q"}:
        if query.lower() == "new":
            thread_id = new_thread()
            print("Started a new conversation.")
        elif query:
            run(graph, query, thread_id)


def new_thread() -> str:
    return uuid.uuid4().hex


def run(graph, query: str, thread_id: str) -> None:
    answer = ""
    config = {"configurable": {"thread_id": thread_id}}
    # stream_mode="updates" yields {node_name: state_update} after each node: handy for seeing the loop.
    for update in graph.stream({"query": query}, config, stream_mode="updates"):
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
        found = f"attempt {change['attempts']}: {len(change['items'])} result(s)"
        if "baseline_items" in change:
            found += f", baseline {len(change['baseline_items'])}"
        return found
    if node == "relax":
        return change["relaxations"][-1]
    if node == "analyze":
        if change.get("analysis_error"):
            return f"skipped: {change['analysis_error']}"
        result = change["analysis"]
        parts = [f"{name} {p['stats'].get('mean', 'n/a')} ({p['stats']['valid_fraction']:.0%} clear)"
                 for name, p in result["periods"].items()]
        if result["change"]:
            mean_change = result["change"].get("mean_change")
            parts.append("change " + ("n/a" if mean_change is None else f"{mean_change:+}"))
        return f"{result['index'].upper()} mean: " + ", ".join(parts) + f" -> {result['provenance']}"
    return "done"


if __name__ == "__main__":
    main()
