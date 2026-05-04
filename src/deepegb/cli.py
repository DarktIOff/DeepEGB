"""Command-line interface for DeepEGB."""
from __future__ import annotations

import json
from pathlib import Path

import click

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv may not be installed in minimal envs
    pass


@click.group()
@click.version_option(package_name="deepegb")
def main() -> None:
    """DeepEGB — model discovery for EGB inflation."""


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
@main.command()
@click.option("--ns", "target_ns", default=0.974, type=float,
              help="Target scalar spectral index n_s.")
@click.option("--ns-sigma", "sigma_ns", default=0.003, type=float)
@click.option("--r", "target_r", default=0.0, type=float,
              help="Target tensor-to-scalar ratio r.")
@click.option("--r-sigma", "sigma_r", default=0.018, type=float)
@click.option("--lnAs", "target_lnAs", default=None, type=float,
              help="Target ln(10^10 A_s); Planck ≈ 3.044.")
@click.option("--alphas", "target_alphas", default=None, type=float,
              help="Target running α_s = dn_s/dlnk.")
@click.option("--nT", "target_nT", default=None, type=float,
              help="Target tensor spectral index n_T.")
@click.option("--cT2", "target_cT2", default=None, type=float,
              help="Target tensor sound speed squared at horizon crossing.")
@click.option("--N", "N_pivot", default=55.0, type=float,
              help="Pivot e-folds before end of inflation.")
@click.option("--niters", "niterations", default=40, type=int)
@click.option("--populations", default=35, type=int)
@click.option("--maxsize", default=25, type=int)
@click.option("--top", default=5, type=int, show_default=True,
              help="How many candidates to print.")
@click.option("--loss", "loss_kind",
              type=click.Choice(["production", "production_gw"]),
              default="production", show_default=True,
              help="`production` (slow-roll closed-form, ~10 ms/eval) or "
                   "`production_gw` (full MS + Ω_GW transfer, ~1 s/eval).")
@click.option("--gw-target", "gw_targets", multiple=True,
              help="Add an Ω_GW target. Format: 'F_HZ:OMEGA_TARGET:SIGMA' "
                   "(repeatable). Example: --gw-target 1e-3:1e-12:5e-13 "
                   "asks for Ω_GW h² ≈ 10⁻¹² at f = 1 mHz (LISA band). "
                   "Implies --loss production_gw.")
@click.option("--gw-band-min", "gw_band_min", default=None,
              help="Penalise Ω_GW dropping below a floor in a frequency band. "
                   "Format: 'F_LO:F_HI:OMEGA_MIN'. Example: "
                   "--gw-band-min 1e-4:1e-1:1e-13. Implies "
                   "--loss production_gw.")
@click.option("--T-reh", "T_reh_GeV", default=1.0e15, type=float,
              help="Reheating temperature in GeV (only matters for the "
                   "transfer-function g_* factor in `production_gw`).")
@click.option("--out", "out_dir", default="runs", type=click.Path())
def search(target_ns, sigma_ns, target_r, sigma_r,
           target_lnAs, target_alphas, target_nT, target_cT2,
           N_pivot, niterations, populations, maxsize, top,
           loss_kind, gw_targets, gw_band_min, T_reh_GeV,
           out_dir):
    """Run a joint symbolic-regression search for V(φ) and ξ(φ)."""
    from rich.console import Console
    from rich.table import Table

    from .search import SearchConfig, run_joint_search

    console = Console()

    # Parse Ω_GW targets
    gw_target_tuples: list[tuple[float, float, float]] = []
    for spec in gw_targets:
        try:
            f_str, om_str, sig_str = spec.split(":")
            gw_target_tuples.append((float(f_str), float(om_str), float(sig_str)))
        except ValueError:
            raise click.BadParameter(
                f"--gw-target must be 'F_HZ:OMEGA:SIGMA' (got {spec!r})")
    band_min = None
    if gw_band_min is not None:
        try:
            f_lo, f_hi, om_min = gw_band_min.split(":")
            band_min = (float(f_lo), float(f_hi), float(om_min))
        except ValueError:
            raise click.BadParameter(
                f"--gw-band-min must be 'F_LO:F_HI:OMEGA_MIN'")
    if (gw_target_tuples or band_min) and loss_kind != "production_gw":
        loss_kind = "production_gw"

    cfg = SearchConfig(
        target_ns=target_ns, sigma_ns=sigma_ns,
        target_r=target_r, sigma_r=sigma_r,
        target_lnAs=target_lnAs, target_alphas=target_alphas,
        target_nT=target_nT, target_cT2=target_cT2,
        N_pivot=N_pivot,
        niterations=niterations, populations=populations,
        maxsize=maxsize,
        loss_kind=loss_kind,
        omega_gw_targets=tuple(gw_target_tuples),
        omega_gw_band_min=band_min,
        T_reh_GeV=T_reh_GeV,
        runs_dir=out_dir,
    )

    console.print(
        f"[bold]Targets[/bold]:  n_s = {target_ns} ± {sigma_ns},  "
        f"r = {target_r} ± {sigma_r},  N = {N_pivot}  "
        f"[dim](loss={loss_kind})[/dim]"
    )
    if any(t is not None for t in (target_lnAs, target_alphas, target_nT, target_cT2)):
        extras = []
        if target_lnAs is not None: extras.append(f"lnAs={target_lnAs}")
        if target_alphas is not None: extras.append(f"α_s={target_alphas}")
        if target_nT is not None: extras.append(f"n_T={target_nT}")
        if target_cT2 is not None: extras.append(f"c_T²={target_cT2}")
        console.print(f"[bold]Extra targets[/bold]: {', '.join(extras)}")

    results = run_joint_search(cfg, progress_cb=lambda s: console.log(s))

    table = Table(title="Top candidates (by χ²)", show_lines=False)
    cols = ("rank", "χ²", "n_s", "r", "n_T", "c_T²", "ε", "V(φ)", "ξ(φ)")
    for col in cols:
        table.add_column(col)
    for i, r in enumerate(results[:top], 1):
        table.add_row(
            str(i),
            f"{r.chi2:.3g}", f"{r.n_s:.4f}", f"{r.r:.4g}",
            f"{r.n_T:.4g}" if r.n_T == r.n_T else "—",
            f"{r.c_T2:.4f}" if r.c_T2 == r.c_T2 else "—",
            f"{r.epsilon:.3g}",
            r.V_expr[:30], r.xi_expr[:30],
        )
    console.print(table)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(out_dir) / "results.json"
    out_json.write_text(json.dumps([r.as_dict() for r in results], default=str, indent=2))
    console.print(f"[green]Saved full results → {out_json}[/green]")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
@main.command()
@click.argument("v_expr")
@click.argument("xi_expr", default="0")
@click.option("--N", "N_pivot", default=55.0, type=float)
def analyze(v_expr: str, xi_expr: str, N_pivot: float) -> None:
    """Analyze a single (V, ξ) model. Example:  deepegb analyze "phi**2" "0"."""
    from rich import print_json
    from .analysis import analyze_egb_model
    out = analyze_egb_model(v_expr, xi_expr, N=N_pivot)
    print_json(data=out, default=str)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
@main.command()
@click.argument("v_expr")
@click.argument("xi_expr", default="0")
@click.option("--N", "N_pivot", default=55.0, type=float)
@click.option("--out", "out_path", default="outputs/egb_diagnostic.png",
              type=click.Path())
def plot(v_expr: str, xi_expr: str, N_pivot: float, out_path: str) -> None:
    """Produce a 6-panel diagnostic plot for a single (V, ξ) model."""
    from .analysis import plot_egb_model
    p = plot_egb_model(v_expr, xi_expr, N=N_pivot, out_path=out_path)
    click.echo(f"Saved → {p}")


@main.command(name="relic-gw")
@click.argument("v_expr")
@click.argument("xi_expr", default="0")
@click.option("--N", "N_pivot", default=55.0, type=float,
              help="Pivot e-folds before end of inflation.")
@click.option("--decades", "n_decades", default=8.0, type=float,
              help="Number of decades of k around the pivot.")
@click.option("--n-k", default=30, type=int,
              help="Number of k samples (more = slower but smoother).")
@click.option("--T-reh", "T_reh_GeV", default=1.0e15, type=float,
              help="Reheating temperature in GeV (controls g_* in transfer).")
@click.option("--out", "out_path", default="outputs/relic_gw.png",
              type=click.Path())
def relic_gw(v_expr: str, xi_expr: str, N_pivot: float,
             n_decades: float, n_k: int, T_reh_GeV: float,
             out_path: str) -> None:
    """Compute and plot the relic GW spectrum Ω_GW(f) h² for an EGB model.

    Uses the full background EOM integration + Mukhanov-Sasaki mode equations
    + radiation/matter-domination transfer function. Overlays experimental
    sensitivity bands (PTA, LISA, DECIGO, ET, LIGO).
    """
    from .analysis import plot_relic_gw_spectrum
    p = plot_relic_gw_spectrum(v_expr, xi_expr, N=N_pivot,
                                n_decades=n_decades, n_k=n_k,
                                T_reh_GeV=T_reh_GeV, out_path=out_path)
    click.echo(f"Saved → {p}")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
@main.command()
@click.option("--provider", default=None,
              type=click.Choice(["local", "anthropic", "openai", "zai"]),
              help="LLM provider (default: $DEEPEGB_PROVIDER or 'local').")
@click.option("--message", "-m", default=None,
              help="Send a single message and exit (non-interactive).")
@click.option("--no-arxiv", is_flag=True, default=False,
              help="Disable the arXiv MCP tools.")
@click.option("--no-rag", is_flag=True, default=False,
              help="Disable the local-RAG `retrieve_literature` tool.")
def chat(provider: str | None, message: str | None,
         no_arxiv: bool, no_rag: bool) -> None:
    """Interactive chat with the DeepEGB agent team."""
    from .agent import run_chat
    run_chat(
        initial_message=message, provider=provider,
        enable_arxiv_mcp=not no_arxiv, enable_local_rag=not no_rag,
    )


# ---------------------------------------------------------------------------
# rag
# ---------------------------------------------------------------------------
@main.group()
def rag() -> None:
    """Manage the local Retrieval-Augmented Generation index."""


@rag.command(name="index")
@click.argument("folder", required=False)
@click.option("--index-dir", default=None, type=click.Path(),
              help="Where to write the index. Default: ~/.deepegb/rag_index.")
@click.option("--model", "embedding_model", default=None,
              help="sentence-transformers model. Default: BAAI/bge-small-en-v1.5.")
@click.option("--max-chars", default=1800, type=int,
              help="Max chunk size in characters.")
@click.option("--overlap", default=200, type=int,
              help="Chunk overlap in characters.")
def rag_index(folder: str | None, index_dir: str | None,
              embedding_model: str | None, max_chars: int, overlap: int) -> None:
    """Build (or rebuild) the local RAG index from a folder of papers.

    FOLDER defaults to $DEEPEGB_RAG_PATH or ~/University/PhD/PhD/papers.
    Walks recursively for PDF / TeX / HTML / Markdown / plain-text files.
    """
    import os
    from .rag import DEFAULT_INDEX_DIR, DEFAULT_MODEL, build_index

    if folder is None:
        folder = os.environ.get("DEEPEGB_RAG_PATH",
                                str(Path.home() / "University/PhD/PhD/papers"))
    idx_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
    model = embedding_model or DEFAULT_MODEL
    meta = build_index(
        Path(folder), index_dir=idx_dir,
        embedding_model=model, max_chars=max_chars, overlap=overlap,
    )
    click.echo(f"\nDone. {meta.n_chunks} chunks indexed at {idx_dir}.")


@rag.command(name="query")
@click.argument("query")
@click.option("--k", default=5, type=int, help="Number of results.")
@click.option("--alpha", default=0.6, type=float,
              help="Weight on dense vs BM25 (1=dense only, 0=BM25 only).")
@click.option("--index-dir", default=None, type=click.Path())
def rag_query(query: str, k: int, alpha: float, index_dir: str | None) -> None:
    """Query the local RAG index. Prints chunks ranked by hybrid score."""
    from .rag import DEFAULT_INDEX_DIR, format_hits_for_llm, hybrid_retrieve

    idx_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
    hits = hybrid_retrieve(query, index_dir=idx_dir, k=k, alpha=alpha)
    if not hits:
        click.echo("No hits.")
        return
    click.echo(format_hits_for_llm(hits))


@rag.command(name="info")
@click.option("--index-dir", default=None, type=click.Path())
def rag_info(index_dir: str | None) -> None:
    """Show metadata about the existing RAG index."""
    import json as _json
    from .rag import DEFAULT_INDEX_DIR, index_exists

    idx_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
    if not index_exists(idx_dir):
        click.echo(f"No index at {idx_dir}. Run `deepegb rag index <folder>` first.")
        return
    meta = (Path(idx_dir) / "metadata.json").read_text()
    click.echo(_json.dumps(_json.loads(meta), indent=2))


if __name__ == "__main__":
    main()
