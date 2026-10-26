"""AutoVV command line. Run without arguments to open the GUI.

    python -m autovv                      # GUI
    python -m autovv train [--source DIR ...] [--no-eval]
    python -m autovv generate FILE [FILE ...] [--material metal] [--mode mer] [--out DIR]
    python -m autovv packs                # list discovered Bedrock resource packs
    python -m autovv scan PACK_DIR        # list a pack's textures and MER status
    python -m autovv ollama-bench MODEL [MODEL ...]   # compare Ollama models on material naming
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .materials import MATERIAL_KEYS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autovv", description="Vanilla-style MERS maps for Minecraft Bedrock.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the graphical interface (default)")

    t = sub.add_parser("train", help="learn from vanilla MERS files and rebuild the knowledge base")
    t.add_argument("--source", action="append", type=Path,
                   help="folder of vanilla textures (repeatable; default: vanilla textures/blocks)")
    t.add_argument("--no-eval", action="store_true", help="skip the hold-out evaluation")

    g = sub.add_parser("generate", help="create MERS maps and texture sets for texture files")
    g.add_argument("files", nargs="+", type=Path)
    g.add_argument("--material", action="append", choices=MATERIAL_KEYS,
                   help="material hint (repeatable); default: auto-detect")
    g.add_argument("--mode", choices=("mers", "mer"), default="mers",
                   help="mers = Vibrant Visuals (default), mer = RTX-compatible 3-channel")
    g.add_argument("--format", choices=("tga", "png"), default="tga")
    g.add_argument("--out", type=Path, help="output folder (default: next to each texture)")
    g.add_argument("--no-noise", action="store_true")
    g.add_argument("--no-edges", action="store_true")
    g.add_argument("--ollama-model", help="ask this Ollama model about unrecognized names")
    g.add_argument("--ollama-host", default="http://localhost:11434")

    sub.add_parser("packs", help="list resource packs in your Minecraft folders")
    s = sub.add_parser("scan", help="list the textures in a resource pack")
    s.add_argument("pack", type=Path)
    s.add_argument("--all", action="store_true", help="include UI/system folders")

    b = sub.add_parser("ollama-bench", help="score Ollama models on naming materials")
    b.add_argument("models", nargs="+")
    b.add_argument("--host", default="http://localhost:11434")

    args = ap.parse_args(argv)
    if args.cmd in (None, "gui"):
        from .gui.app import run
        return run()
    if args.cmd == "train":
        from .trainer import train
        train(args.source, evaluate_folds=0 if args.no_eval else 5)
        return 0
    if args.cmd == "generate":
        return _generate(args)
    if args.cmd == "packs":
        from .pack import discover_packs
        for p in discover_packs():
            print(f"{p.display}\n    {p.path}")
        return 0
    if args.cmd == "scan":
        from .pack import scan_pack
        for folder, items in scan_pack(args.pack, args.all).folders().items():
            print(f"{folder}/")
            for t in items:
                print(f"    {t.name:40s} {'[has MER]' if t.has_mer else ''}")
        return 0
    if args.cmd == "ollama-bench":
        from .ollama_client import OllamaClient, benchmark
        if not OllamaClient(args.host).available():
            print(f"Ollama is not reachable at {args.host}. Start Ollama and pull the models first.")
            return 1
        results = []
        for m in args.models:
            print(f"{m}:")
            results.append(benchmark(OllamaClient(args.host, m)))
        print()
        print("model                          accuracy   s/name")
        for r in sorted(results, key=lambda r: (-r["accuracy"], r["seconds_per_name"])):
            print(f"{r['model']:30s} {r['accuracy']:8.0%} {r['seconds_per_name']:8.2f}")
        return 0
    return 1


def _generate(args) -> int:
    from .knowledge import KnowledgeBase
    from .ollama_client import OllamaClient
    from .pipeline import OutputOptions, decide_material, process_file

    kb = KnowledgeBase.load()
    opts = OutputOptions(args.mode, args.format, not args.no_noise, not args.no_edges)
    ollama = OllamaClient(args.ollama_host, args.ollama_model) if args.ollama_model else None
    for f in args.files:
        decision = decide_material(f.stem, f.parent.name, args.material, ollama)
        r = process_file(kb, f, args.out, decision, f.parent.name, opts)
        print(f"{f.name} -> {r.map_path.name}" + (f" + {r.texture_set_path.name}" if r.texture_set_path else ""))
        print(r.result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
