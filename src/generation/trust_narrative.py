"""Trust Narrative generator explaining RiskPath retrieval and repair decisions."""

from typing import Optional, Dict, Any
from src.graph.schema import ReasoningPath
from src.repair.recovery import RepairResult


class TrustNarrativeGenerator:
    """Generates natural language explainability narratives justifying retrieval and repair actions."""

    @staticmethod
    def generate_narrative(
        query: str,
        selected_path: ReasoningPath,
        repair_result: Optional[RepairResult] = None,
        method_name: str = "RiskPath",
    ) -> str:
        """Constructs an auditable narrative explaining why specific facts were trusted or rejected."""
        if not selected_path.edges:
            return f"[{method_name}] No viable reasoning path could be discovered for the query."

        lines = [f"### Trust Narrative ({method_name})"]
        lines.append(f"**Query:** \"{query}\"")
        lines.append(f"**Selected Reasoning Path:** `{selected_path.linearize()}`")
        lines.append(f"**Path Status:** `{selected_path.status}` | **Max PWRS Risk:** `{selected_path.max_risk:.4f}`")

        if repair_result and repair_result.status == "REPAIRED":
            flagged = repair_result.flagged_edge
            flagged_str = flagged.to_triple_string() if flagged else "Unknown edge"
            lines.append("\n**Risk Detection & Recovery Event:**")
            lines.append(
                f"- RiskPath identified a critical vulnerability in the primary candidate path on triple `{flagged_str}`."
            )
            lines.append(
                f"- While the triple seemed topically relevant, its base reliability was low ({flagged.base_reliability:.2f}) "
                f"and its causal answer influence was high ({flagged.causal_importance:.2f}), resulting in high PWRS ({repair_result.original_pwrs:.4f})."
            )
            lines.append(
                f"- **Mitigation Action:** The high-risk edge was pruned, and the system dynamically routed through an alternative valid reasoning path (`{selected_path.linearize()}`)."
            )
            lines.append(
                f"- **Result:** Answer fidelity was recovered without exposing the generative model to corrupted evidence."
            )
        elif selected_path.status == "FLAGGED":
            lines.append(
                "\n**Risk Warning:** The selected path contains facts exceeding the safety threshold, but no viable alternative was discovered. Answer should be audited."
            )
        else:
            lines.append(
                "\n**Verification Summary:** All facts on this reasoning chain were verified with sufficient provenance support and low propagation risk."
            )

        return "\n".join(lines)
