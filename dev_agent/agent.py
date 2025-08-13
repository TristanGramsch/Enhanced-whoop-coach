from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def collect_context(repo_root: Path) -> Dict[str, Any]:
    outputs = repo_root / "shared" / "outputs"
    ctx: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "latest_files": [],
        "notes": [],
    }
    try:
        for p in [outputs / "training_metrics.json", outputs / "intelligence_summary.json", outputs / "agent_forecast.json", outputs / "predictions_registry.csv"]:
            if p.exists():
                ctx["latest_files"].append(str(p))
    except Exception:
        pass
    return ctx


def run_dev_debrief(outputs_dir: str, repo_root: str) -> Dict[str, Any]:
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root)
    ctx = collect_context(root)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Fallback: simple debrief stub
        report = {
            "date": datetime.utcnow().date().isoformat(),
            "summary": "No LLM key provided; no automated development today.",
            "changes_made": [],
            "reasons": ["OPENAI_API_KEY missing"],
            "test_results": None,
            "next_actions": ["Provide OPENAI_API_KEY to enable GPT-5 debrief"],
        }
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            system = (
                "You are a software engineer maintaining an HRV forecasting repo. Summarize what should be improved next based on the current artifacts,"
                " with concrete reasons and a short prioritized list. Output strict JSON with keys: summary, changes_made (array of strings), reasons (array), next_actions (array)."
            )
            user_content = json.dumps({"context": ctx})
            completion = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            llm_json = json.loads(content)
            report = {
                "date": datetime.utcnow().date().isoformat(),
                **llm_json,
            }
        except Exception as e:
            report = {
                "date": datetime.utcnow().date().isoformat(),
                "summary": f"LLM debrief failed: {e}",
                "changes_made": [],
                "reasons": ["LLM call error"],
                "next_actions": ["Retry tomorrow"],
            }

    out_path = out_dir / "dev_debrief.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return {"dev_debrief_path": str(out_path)}


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    outputs_dir = os.environ.get("OUTPUTS_DIR", str(repo / "shared" / "outputs"))
    artifacts = run_dev_debrief(outputs_dir, str(repo))
    print(json.dumps(artifacts, indent=2)) 