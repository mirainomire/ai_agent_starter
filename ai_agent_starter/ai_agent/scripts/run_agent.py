#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run AI Security Testing Agent - Local Development Mode
"""
import argparse
import os
import sys
from pathlib import Path
import traceback
from datetime import datetime, timezone

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # python-dotenv not installed; run: pip install python-dotenv
from rich.console import Console
from rich.panel import Panel

# ✅ Add project root to path
# run_agent.py is in ai_agent/scripts/, so parent.parent is ai_agent_starter root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Go up from scripts/ -> ai_agent/ -> ai_agent_starter/
sys.path.insert(0, str(project_root))

# Search several repository/CWD locations for .env. The example template is
# never loaded at runtime because placeholder credentials are not secrets.
if load_dotenv is not None:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "ai_agent_starter" / ".env",
            repo_root / ".env",
            repo_root.parent / ".env",
            Path.cwd() / ".env",
        ]
        loaded = False
        for env_path in candidates:
            if env_path.exists():
                # Explicit terminal/CI environment values take precedence over
                # repository-local defaults.
                load_dotenv(dotenv_path=env_path, override=False)
                loaded = True
                break
        if not loaded:
            load_dotenv()
    except Exception:
        load_dotenv()
else:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv", file=sys.stderr)

from ai_agent.core.orchestrator import AgentOrchestrator

console = Console()

def _write_agent_log(message: str) -> None:
    """Persist runtime errors to ai_agent/runs/logs/agent.log."""
    try:
        log_dir = project_root / "ai_agent" / "runs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "agent.log"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        # Logging should never break runner behavior
        pass

def render_banner():
    """Render a big ASCII banner and subtitle (configurable via env).
    BANNER_TITLE, BANNER_SUBTITLE, BANNER_FONT can be set in .env
    """
    title = os.getenv("BANNER_TITLE", "AI Security Testing Agent")
    subtitle = os.getenv("BANNER_SUBTITLE", "Broken Access Control (IDOR/BOLA) Orchestrator")
    font = os.getenv("BANNER_FONT", "Slant")
    try:
        from pyfiglet import Figlet
        fig = Figlet(font=font)
        ascii_title = fig.renderText(title)
        console.print(f"[bold cyan]{ascii_title}[/bold cyan]")
    except Exception:
        # Fallback to a simple panel if pyfiglet not available
        console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]", border_style="cyan"))
        return
    # Subtitle panel below the ASCII art
    console.print(Panel.fit(f"[dim]{subtitle}[/dim]", border_style="cyan"))

def chooseRunMode() -> str:
    """Return "1" (deterministic) or "2" (LLM Gemini) from --mode or an interactive prompt."""
    parser = argparse.ArgumentParser(description="Run the BAC testing agent")
    parser.add_argument("--mode", choices=["1", "2"],
                        help="1 = deterministic (no LLM), 2 = LLM Gemini (requires GEMINI_API_KEY)")
    args = parser.parse_args()
    if args.mode:
        return args.mode
    console.print("\nChoose a mode:")
    console.print("  [bold]1[/bold]) Deterministic (no Gemini API key)")
    console.print("  [bold]2[/bold]) LLM Gemini (requires GEMINI_API_KEY)")
    try:
        choice = input("Mode [1/2]: ").strip()
    except EOFError:
        choice = ""
    if choice not in ("1", "2"):
        console.print("❌ Invalid choice. Enter 1 or 2, or run with --mode 1 / --mode 2.", style="red")
        sys.exit(1)
    return choice


def main():
    render_banner()

    run_choice = chooseRunMode()
    if run_choice == "1":
        os.environ["AGENT_MODE"] = "deterministic"
        os.environ["LLM_PROVIDER"] = "none"
        console.print("✅ Mode 1: Deterministic (no LLM)", style="green")
    else:
        if not (os.getenv("GEMINI_API_KEY") or "").strip():
            console.print("❌ Mode 2 requires GEMINI_API_KEY, but it was not found in .env.", style="red")
            sys.exit(1)
        os.environ["AGENT_MODE"] = "llm"
        os.environ["LLM_PROVIDER"] = "gemini"
        console.print(f"✅ Mode 2: LLM Gemini ({os.getenv('GEMINI_MODEL') or 'default model'})", style="green")

    console.print("\n🚀 Starting agent...\n")
    
    try:
        # ✅ Run orchestrator
        agent = AgentOrchestrator()
        results = agent.run()
        
        # Print summary (clean output)
        console.print("\nTesting completed!", style="green bold")
        total = int(results.get("total_tests", 0) or 0)
        generated = int(results.get("generated_cases", 0) or 0)
        evaluated = int(results.get("evaluated_cases", 0) or 0)
        vulns = int(results.get("vulnerabilities", 0) or 0)
        console.print(f"Generated cases: {generated}")
        coverage = results.get('operation_coverage') or {}
        console.print(f"OAS operations: {coverage.get('operations', 0)}; accounts: {coverage.get('accounts', 0)}")
        console.print(f"Operation/account coverage: {coverage.get('covered_pairs', 0)}/{coverage.get('operation_account_pairs', 0)}")
        for account, count in (results.get('cases_by_account') or {}).items():
            console.print(f"  {account}: {count} cases", markup=False)
        console.print(f"Result records: {total}")
        console.print(f"Evaluated cases: {evaluated}")
        console.print(f"Vulnerabilities found: {vulns}", style="red" if vulns > 0 else "green")
        m = results.get("metrics") or {"precision":0.0,"recall":0.0,"f1":0.0,"accuracy":0.0}
        console.print(f"Metrics -> precision: {m.get('precision',0.0)}, recall: {m.get('recall',0.0)}, f1: {m.get('f1',0.0)}, accuracy: {m.get('accuracy',0.0)}")
        console.print(f"Report saved to: {results.get('report_path','')}" )
    except KeyboardInterrupt:
        console.print("\n⚠️  Agent stopped by user", style="yellow")
        sys.exit(0)
    except Exception as e:
        err_text = str(e)
        tb = traceback.format_exc()
        _write_agent_log(f"ERROR: {err_text}\n{tb}")
        console.print(f"\n❌ Error: {err_text}", style="red")
        if "getaddrinfo failed" in err_text.lower():
            console.print("[yellow]Hint: DNS/network to the LLM provider failed. Check internet/proxy/firewall or disable strict summary.[/yellow]")
        console.print("\n[dim]Check logs at: ai_agent/runs/logs/agent.log[/dim]")
        sys.exit(1)

if __name__ == "__main__":
    main()
