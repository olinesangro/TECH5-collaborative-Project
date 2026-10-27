"""Transparent recommendation generator distinguishing evidence from advice.

This module implements the decision-support engine required by Section 9 of the
TECH5 specification, providing inspectable recommendations with explicit
criteria, mathematical evidence, and policy-oriented advice.
"""

import collections.abc
import dataclasses
from typing import Any

from app.domain import models


@dataclasses.dataclass
class RecommendationItem:
    """An individual decision recommendation.

    Attributes:
        priority_focus: The organizational objective prioritized (e.g.
            Student Satisfaction, Project Skill Delivery, Egalitarian Balance).
        recommended_solution_id: Identifier of the top solution for this focus.
        recommended_alternative_number: 1-based index of recommended solution.
        criteria_used: Explicit list of analytical metrics evaluated.
        evidence: Objective calculated facts supporting the recommendation.
        advice: Practical normative advice describing operational trade-offs.
        notes: Decision notes, e.g. ties or a solution that wins several
            priorities, so the user sees how the pick was reached.
    """

    priority_focus: str
    recommended_solution_id: str
    recommended_alternative_number: int
    criteria_used: list[str]
    evidence: list[str]
    advice: str
    notes: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializes recommendation item to a dictionary.

        Returns:
            Dictionary containing recommendation fields.
        """
        return {
            "priority_focus": self.priority_focus,
            "recommended_solution_id": self.recommended_solution_id,
            "recommended_alternative_number": (
                self.recommended_alternative_number
            ),
            "criteria_used": list(self.criteria_used),
            "evidence": list(self.evidence),
            "advice": self.advice,
            "notes": list(self.notes),
        }


def _fairness(solution: models.Solution) -> dict[str, Any]:
    """Returns the fairness indicators of a solution (empty if missing)."""
    return solution.indicators.get("fairness", {})


def _efficiency(solution: models.Solution) -> dict[str, Any]:
    """Returns the efficiency indicators of a solution (empty if missing)."""
    return solution.indicators.get("efficiency", {})


def _preference_key(solution: models.Solution) -> tuple[Any, ...]:
    """Ranking key for student preference satisfaction (lower is better)."""
    fair = _fairness(solution)
    return (
        fair.get("unranked_count", 999),
        fair.get("mean_rank", 999.0),
        fair.get("worst_rank", 999),
    )


def _skill_key(solution: models.Solution) -> tuple[Any, ...]:
    """Ranking key for project skill fit (lower is better)."""
    eff = _efficiency(solution)
    return (
        -eff.get("mean_skill_fit", -1.0),
        eff.get("unmet_soft_requirements_count", 999),
    )


def _balanced_key(solution: models.Solution) -> tuple[Any, ...]:
    """Ranking key for egalitarian balance (lower is better)."""
    fair = _fairness(solution)
    return (
        fair.get("rank_gini", 1.0),
        fair.get("rank_variance", 999.0),
        fair.get("mean_rank", 999.0),
    )


def _pick_best(
    solutions: list[models.Solution],
    key: collections.abc.Callable[[models.Solution], tuple[Any, ...]],
) -> tuple[models.Solution, list[models.Solution]]:
    """Finds the best solution under a ranking key and any exact ties.

    Args:
        solutions: Candidate solutions (non-empty).
        key: Function returning a sortable key where lower is better.

    Returns:
        A tuple (best, tied). best is the lowest-numbered alternative among
        the top scorers, and tied holds every solution with the same score.
    """
    ordered = sorted(solutions, key=lambda s: s.alternative_number)
    best = min(ordered, key=key)
    tied = [s for s in ordered if key(s) == key(best)]
    return best, tied


def _tie_note(tied: list[models.Solution]) -> list[str]:
    """Explains a tie between solutions, if there is one."""
    if len(tied) < 2:
        return []
    numbers = ", ".join(str(s.alternative_number) for s in tied)
    return [
        f"Tie: Solutions {numbers} score identically on these criteria. "
        f"Solution {tied[0].alternative_number} is shown because it was "
        "generated first; the others are equally good choices here."
    ]


def _add_shared_winner_notes(
    recommendations: list[RecommendationItem], solution_count: int
) -> None:
    """Adds notes when one solution is recommended for several priorities.

    Args:
        recommendations: Recommendations to annotate in place.
        solution_count: Number of alternative solutions in the run.
    """
    if solution_count == 1:
        for rec in recommendations:
            rec.notes.append(
                "Only one feasible alternative was generated, so every "
                "priority recommends it. There is no trade-off to choose."
            )
        return

    for rec in recommendations:
        others = [
            r.priority_focus
            for r in recommendations
            if r is not rec
            and r.recommended_solution_id == rec.recommended_solution_id
        ]
        if len(others) == len(recommendations) - 1:
            rec.notes.append(
                f"Solution {rec.recommended_alternative_number} is the top "
                "choice under every priority, so no trade-off is needed."
            )
        elif others:
            rec.notes.append(
                f"Solution {rec.recommended_alternative_number} is also the "
                f"top choice for: {'; '.join(others)}."
            )


def generate_recommendations(
    solutions: list[models.Solution],
) -> list[RecommendationItem]:
    """Generates policy-oriented recommendations from alternative solutions.

    For each of three administrative priorities, the best alternative is
    picked with an explicit ranking key. The evidence lists the calculated
    numbers behind the pick, the advice is kept separate, and notes flag
    ties and solutions that win several priorities.

    Args:
        solutions: Feasible alternative solutions from an assignment run.

    Returns:
        List of structured RecommendationItem instances.
    """
    if not solutions:
        return []

    recommendations: list[RecommendationItem] = []

    # Priority 1: Maximizing Student Satisfaction
    best_pref_sol, tied = _pick_best(solutions, _preference_key)
    pref_fairness = _fairness(best_pref_sol)
    pref_efficiency = _efficiency(best_pref_sol)

    recommendations.append(
        RecommendationItem(
            priority_focus="Student Preference Maximization",
            recommended_solution_id=best_pref_sol.id,
            recommended_alternative_number=best_pref_sol.alternative_number,
            criteria_used=[
                "Count of students assigned outside stated preferences "
                "(lower is better)",
                "Average student preference rank (lower is better)",
                "Worst individual student rank (lower is better)",
            ],
            evidence=[
                f"Achieves average preference rank of "
                f"{pref_fairness.get('mean_rank', 'N/A')}.",
                f"Assigned {pref_fairness.get('unranked_count', 0)} students "
                "outside stated preferences.",
                f"Worst individual student rank is "
                f"{pref_fairness.get('worst_rank', 'N/A')}.",
                f"Maintains average skill-fit score of "
                f"{pref_efficiency.get('mean_skill_fit', 0.0):.2f}.",
            ],
            advice=(
                f"Adopt Solution {best_pref_sol.alternative_number} if "
                "student satisfaction, programme morale, and minimizing "
                "student appeals are the primary administrative goals."
            ),
            notes=_tie_note(tied),
        )
    )

    # Priority 2: Maximizing Project Skill Fit & Deliverable Quality
    best_skill_sol, tied = _pick_best(solutions, _skill_key)
    skill_efficiency = _efficiency(best_skill_sol)
    skill_fairness = _fairness(best_skill_sol)

    recommendations.append(
        RecommendationItem(
            priority_focus="Project Skill Quality & Delivery Assurance",
            recommended_solution_id=best_skill_sol.id,
            recommended_alternative_number=best_skill_sol.alternative_number,
            criteria_used=[
                "Average student-project skill fit score (higher is better)",
                "Total unmet soft skill requirements (lower is better)",
            ],
            evidence=[
                f"Highest average skill fit score of "
                f"{skill_efficiency.get('mean_skill_fit', 0.0):.2f}.",
                f"Unmet soft requirements count: "
                f"{skill_efficiency.get('unmet_soft_requirements_count', 0)}.",
                f"Average student preference rank is "
                f"{skill_fairness.get('mean_rank', 'N/A')}.",
            ],
            advice=(
                f"Adopt Solution {best_skill_sol.alternative_number} when "
                "partner organizations or client project deliverables demand "
                "rigorous technical and domain capabilities."
            ),
            notes=_tie_note(tied),
        )
    )

    # Priority 3: Egalitarian Fairness & Balanced Trade-off
    best_balanced_sol, tied = _pick_best(solutions, _balanced_key)
    bal_fairness = _fairness(best_balanced_sol)
    bal_efficiency = _efficiency(best_balanced_sol)

    recommendations.append(
        RecommendationItem(
            priority_focus="Balanced & Egalitarian Allocation",
            recommended_solution_id=best_balanced_sol.id,
            recommended_alternative_number=best_balanced_sol.alternative_number,
            criteria_used=[
                "Rank Gini coefficient (equality of outcome distribution, "
                "lower is better)",
                "Rank variance across students (lower is better)",
                "Average preference rank, used only to break ties",
            ],
            evidence=[
                f"Lowest rank disparity (Gini coefficient: "
                f"{bal_fairness.get('rank_gini', 'N/A')}).",
                f"Rank variance: {bal_fairness.get('rank_variance', 'N/A')}.",
                f"Average rank: {bal_fairness.get('mean_rank', 'N/A')}.",
                f"Average skill fit (shown for context, not used in the "
                f"choice): {bal_efficiency.get('mean_skill_fit', 0.0):.2f}.",
            ],
            advice=(
                f"Adopt Solution {best_balanced_sol.alternative_number} if "
                "the academic committee seeks a fair compromise that avoids "
                "extreme individual disadvantage."
            ),
            notes=_tie_note(tied),
        )
    )

    _add_shared_winner_notes(recommendations, len(solutions))
    return recommendations
