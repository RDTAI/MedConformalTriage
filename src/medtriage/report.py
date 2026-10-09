"""Report exporters for review boards and model cards."""
import csv, json

def render_markdown(report):
    value = lambda key: "n/a" if report.get(key) is None else f"{report[key]:.4f}"
    lines = ["# Medical Conformal Triage Report", "", "## Workload and safety", "", f"- Cases: {report['n']}", f"- Auto-processing rate: {value('auto_rate')}", f"- Auto-region error rate: {value('auto_error_rate')}", f"- Human-review rate: {value('review_rate')}", f"- Rejection rate: {value('reject_rate')}", f"- Marginal coverage: {value('coverage')}", f"- Mean prediction-set size: {value('mean_set_size')}", f"- High-risk error capture: {value('high_risk_error_capture')}"]
    if "worst_group_coverage" in report:
        lines += [f"- Worst-group coverage: {value('worst_group_coverage')}"]
    return "\n".join(lines) + "\n"

def write_outputs(report, output_prefix):
    with open(f"{output_prefix}.json", "w") as f: json.dump(report, f, indent=2)
    rows = [{"metric": k, "value": v} for k, v in report.items() if isinstance(v, (int, float)) or v is None]
    with open(f"{output_prefix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"]); writer.writeheader(); writer.writerows(rows)
    with open(f"{output_prefix}.md", "w") as f: f.write(render_markdown(report))
