"""Unit tests for the transparent recommendation logic (Team A)."""

from app.domain import models
from app.engine import recommendations


def _solution(
    number: int,
    mean_rank: float,
    gini: float,
    skill_fit: float,
    unranked: int = 0,
) -> models.Solution:
    """Builds a solution with only the indicators the logic reads."""
    return models.Solution(
        id=f"sol{number}",
        run_id="run1",
        alternative_number=number,
        strategy_name="TEST",
        indicators={
            "fairness": {
                "mean_rank": mean_rank,
                "unranked_count": unranked,
                "worst_rank": 3,
                "rank_gini": gini,
                "rank_variance": gini,
            },
            "efficiency": {
                "mean_skill_fit": skill_fit,
                "unmet_soft_requirements_count": 0,
            },
        },
    )


def _picks(recs: list[recommendations.RecommendationItem]) -> list[int]:
    """Returns the recommended alternative number per priority."""
    return [r.recommended_alternative_number for r in recs]


def test_empty_input_gives_no_recommendations() -> None:
    """No solutions means no recommendations."""
    assert recommendations.generate_recommendations([]) == []


def test_each_priority_picks_its_best_solution() -> None:
    """Preference, skill fit and balance each pick a different winner."""
    sols = [
        _solution(1, mean_rank=1.5, gini=0.20, skill_fit=0.70),
        _solution(2, mean_rank=2.0, gini=0.25, skill_fit=0.90),
        _solution(3, mean_rank=1.8, gini=0.10, skill_fit=0.75),
    ]
    recs = recommendations.generate_recommendations(sols)

    assert _picks(recs) == [1, 2, 3]
    for rec in recs:
        assert rec.criteria_used
        assert rec.evidence
        assert rec.advice
        assert rec.notes == []


def test_unranked_students_outweigh_average_rank() -> None:
    """Fewer students outside their preferences wins before mean rank."""
    sols = [
        _solution(1, mean_rank=1.2, gini=0.2, skill_fit=0.8, unranked=2),
        _solution(2, mean_rank=2.5, gini=0.2, skill_fit=0.8, unranked=0),
    ]
    recs = recommendations.generate_recommendations(sols)
    assert recs[0].recommended_alternative_number == 2


def test_tie_is_reported() -> None:
    """Identical scores pick the first alternative and say it is a tie."""
    sols = [
        _solution(2, mean_rank=1.8, gini=0.1, skill_fit=0.8),
        _solution(1, mean_rank=1.8, gini=0.1, skill_fit=0.8),
    ]
    recs = recommendations.generate_recommendations(sols)

    assert _picks(recs) == [1, 1, 1]
    assert any("Tie: Solutions 1, 2" in n for n in recs[0].notes)


def test_dominant_solution_is_flagged() -> None:
    """A solution best on every priority is flagged as such."""
    sols = [
        _solution(1, mean_rank=1.5, gini=0.10, skill_fit=0.90),
        _solution(2, mean_rank=2.0, gini=0.30, skill_fit=0.70),
    ]
    recs = recommendations.generate_recommendations(sols)

    assert _picks(recs) == [1, 1, 1]
    for rec in recs:
        assert any("every priority" in n for n in rec.notes)


def test_shared_winner_for_two_priorities() -> None:
    """A solution winning two of three priorities names the other one."""
    sols = [
        _solution(1, mean_rank=1.5, gini=0.10, skill_fit=0.70),
        _solution(2, mean_rank=2.0, gini=0.30, skill_fit=0.90),
    ]
    recs = recommendations.generate_recommendations(sols)

    assert _picks(recs) == [1, 2, 1]
    assert any("Balanced" in n for n in recs[0].notes)
    assert recs[1].notes == []


def test_single_solution_is_explained() -> None:
    """With one alternative, every priority says there is no choice."""
    recs = recommendations.generate_recommendations(
        [_solution(1, mean_rank=2.0, gini=0.2, skill_fit=0.8)]
    )
    for rec in recs:
        assert any("Only one feasible alternative" in n for n in rec.notes)


def test_missing_indicators_do_not_crash() -> None:
    """Solutions without indicators still produce recommendations."""
    sol = models.Solution(
        id="s1", run_id="r1", alternative_number=1, strategy_name="X"
    )
    recs = recommendations.generate_recommendations([sol])
    assert len(recs) == 3
