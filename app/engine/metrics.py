"""Calculators for feasibility, efficiency, fairness, and trade-offs.

This module provides analytical functions to evaluate solutions according to
the requirements in Section 7 of the TECH5 specification.

Team B implementation: student-project skill fit, Gini inequality of
assigned ranks, solution indicators, and side-by-side trade-off comparisons.
"""

import collections
from typing import Any

from app.domain import models


def calculate_assignment_skill_fit(
    student: models.Student,
    project: models.Project,
) -> tuple[float, list[str]]:
    """Calculates skill fit score and unmet requirements for a student-project pair.

    In compliance with Section 7.2 of the TECH5 specification, skill fit
    evaluates how well the candidate student satisfies the required technical
    skills of the target project.

    Args:
        student: The student under evaluation.
        project: The project under evaluation.

    Returns:
        A tuple of (skill_fit_score [0.0, 1.0], list_of_unmet_skill_ids).
    """

    if not project.requirements:
        return 1.0, []

    partial_fits: list[float] = []
    unmet: list[str] = []
    for req in project.requirements:
        level = student.get_skill_level(req.skill_id)
        if req.minimum_level <= 0:
            partial_fits.append(1.0)
            continue
        partial_fits.append(min(level / req.minimum_level, 1.0))
        if level < req.minimum_level:
            unmet.append(req.skill_id)

    score = sum(partial_fits) / len(partial_fits)
    return round(score, 4), unmet

def calculate_gini(values: list[float]) -> float:
    """Calculates the Gini coefficient of inequality for non-negative values.

    In compliance with Section 7.3 of the TECH5 specification, the Gini
    coefficient measures the inequality/fairness in student preference rank
    distribution (0.0 represents perfect equality, 1.0 represents extreme inequality).

    Args:
        values: Non-negative numerical series (e.g. assigned preference ranks).

    Returns:
        The Gini coefficient in [0.0, 1.0].
    """
    if not values:
        return 0.0
    if any(v < 0 for v in values):
        raise ValueError("Gini coefficient requires non-negative values.")

    n = len(values)
    total = float(sum(values))
    if total == 0.0:
        return 0.0

    mean = total / n
    abs_diff_sum = sum(abs(x_i - x_j) for x_i in values for x_j in values)
    return round(abs_diff_sum / (2.0 * n * n * mean), 4)


def compute_solution_indicators(
    students: list[models.Student],
    projects: list[models.Project],
    assignments: list[models.Assignment],
    fixed_assignments: list[models.FixedAssignment] | None = None,
) -> dict[str, Any]:
    """Computes comprehensive feasibility, efficiency, and fairness indicators.

    Evaluates a solution across the four dimensions defined in the specification:
    1. Feasibility: Exact student count, capacity bounds, fixed assignments.
    2. Efficiency: Mean skill-fit score, unmet soft requirements count.
    3. Fairness: Rank distribution, mean rank, worst rank, unranked count, Gini.
    4. Capacity: Total assigned, total capacity, overall utilization, per-project.

    Args:
        students: Complete cohort of students.
        projects: Catalog of semester projects.
        assignments: List of student-to-project allocations.
        fixed_assignments: Optional list of locked pre-allocations.

    Returns:
        Dictionary containing structured indicators and diagnostic metrics:
        {
            "feasibility": {"is_feasible": bool, "violations": list[str], ...},
            "efficiency": {"mean_skill_fit": float, ...},
            "fairness": {"mean_rank": float, "worst_rank": int, "rank_gini": float, ...},
            "capacity": {"overall_utilization_pct": float, "projects": dict, ...}
        }
    """
    fixed_assignments = fixed_assignments or []
    student_map = {s.id: s for s in students}
    project_map = {p.id: p for p in projects}
    violations: list[str] = []

    # 1. Feasibility: completeness and uniqueness of allocation.
    per_student = collections.Counter(a.student_id for a in assignments)
    for s in students:
        count = per_student.get(s.id, 0)
        if count == 0:
            violations.append(f"Student {s.id} is not assigned.")
        elif count > 1:
            violations.append(
                f"Student {s.id} is assigned {count} times."
            )
    for a in assignments:
        if a.student_id not in student_map:
            violations.append(f"Unknown student {a.student_id} assigned.")
        if a.project_id not in project_map:
            violations.append(f"Unknown project {a.project_id} assigned.")

    # Capacity bounds l_p <= headcount_p <= u_p.
    per_project = collections.Counter(a.project_id for a in assignments)
    for p in projects:
        count = per_project.get(p.id, 0)
        if count > p.max_capacity:
            violations.append(
                f"Project {p.id} over capacity ({count} > {p.max_capacity})."
            )
        if count < p.min_capacity:
            violations.append(
                f"Project {p.id} under minimum ({count} < {p.min_capacity})."
            )

    # Fixed locks must be preserved.
    allocated = {a.student_id: a.project_id for a in assignments}
    for fa in fixed_assignments:
        if allocated.get(fa.student_id) != fa.project_id:
            violations.append(
                f"Fixed assignment {fa.student_id} -> {fa.project_id} "
                "not preserved."
            )

    # Mandatory (hard) skill coverage: every staffed project needs at least
    # one assigned student meeting each mandatory requirement. Individual
    # shortfalls on mandatory skills are reported but are not violations.
    members: dict[str, list[models.Student]] = collections.defaultdict(list)
    for a in assignments:
        student = student_map.get(a.student_id)
        if student is not None:
            members[a.project_id].append(student)

    mandatory_unmet = 0
    mandatory_total = 0
    mandatory_individual_shortfalls = 0
    soft_unmet = 0
    for p in projects:
        team = members.get(p.id, [])
        for req in p.requirements:
            short = [
                s for s in team
                if s.get_skill_level(req.skill_id) < req.minimum_level
            ]
            if not req.is_mandatory:
                soft_unmet += len(short)
                continue
            if not team:
                continue
            mandatory_total += 1
            mandatory_individual_shortfalls += len(short)
            if len(short) == len(team):
                mandatory_unmet += 1
                violations.append(
                    f"Project {p.id} has no assigned student meeting "
                    f"mandatory skill {req.skill_id} "
                    f"(level {req.minimum_level})."
                )

    # 2. Efficiency.
    n = len(assignments)
    fit_sum = sum(a.skill_fit_score for a in assignments)
    mean_fit = fit_sum / n if n else 0.0

    # 3. Fairness.
    max_prefs = max((len(s.preferences) for s in students), default=0)
    penalty_rank = max_prefs + 1
    effective_ranks: list[int] = []
    ranked_only: list[int] = []
    unranked = 0
    for a in assignments:
        rank = a.preference_rank
        student = student_map.get(a.student_id)
        if rank is None and student is not None:
            rank = student.get_preference_rank(a.project_id)
        if rank is None:
            unranked += 1
            effective_ranks.append(penalty_rank)
        else:
            ranked_only.append(rank)
            effective_ranks.append(rank)

    if effective_ranks:
        mean_rank = sum(effective_ranks) / len(effective_ranks)
        rank_variance = sum(
            (r - mean_rank) ** 2 for r in effective_ranks
        ) / len(effective_ranks)
        worst_rank = max(effective_ranks)
    else:
        mean_rank, rank_variance, worst_rank = 0.0, 0.0, 0

    rank_counts = collections.Counter(ranked_only)
    rank_distribution = {
        f"rank_{r}": rank_counts[r] for r in sorted(rank_counts)
    }

    # 4. Capacity utilization.
    total_capacity = sum(p.max_capacity for p in projects)
    project_util: dict[str, dict[str, Any]] = {}
    for p in projects:
        count = per_project.get(p.id, 0)
        project_util[p.id] = {
            "assigned_count": count,
            "min_capacity": p.min_capacity,
            "max_capacity": p.max_capacity,
            "utilization_pct": round(
                count / p.max_capacity * 100.0, 1
            ) if p.max_capacity > 0 else 0.0,
        }

    return {
        "feasibility": {
            "is_feasible": not violations,
            "violations": violations,
            "assigned_count": len(allocated),
            "total_students": len(students),
            "mandatory_unmet_count": mandatory_unmet,
            "mandatory_requirements_covered": mandatory_total - mandatory_unmet,
            "mandatory_requirements_total": mandatory_total,
            "mandatory_individual_shortfalls": mandatory_individual_shortfalls,
        },
        "efficiency": {
            "mean_skill_fit": round(mean_fit, 4),
            "unmet_soft_requirements_count": soft_unmet,
            "total_skill_fit_sum": round(fit_sum, 4),
        },
        "fairness": {
            "rank_distribution": rank_distribution,
            "unranked_count": unranked,
            "unranked_pct": round(unranked / n * 100.0, 1) if n else 0.0,
            "worst_rank": worst_rank,
            "mean_rank": round(mean_rank, 4),
            "rank_variance": round(rank_variance, 4),
            "rank_gini": calculate_gini([float(r) for r in effective_ranks]),
        },
        "capacity": {
            "total_assigned": n,
            "total_capacity": total_capacity,
            "overall_utilization_pct": round(
                n / total_capacity * 100.0, 1
            ) if total_capacity > 0 else 0.0,
            "projects": project_util,
        },
    }

def _section_value(
    solution: models.Solution, section: str, key: str
) -> float | None:
    """Reads one numeric indicator from a solution, or None if absent."""
    value = solution.indicators.get(section, {}).get(key)
    return float(value) if value is not None else None


def _fallback_means(solution: models.Solution) -> tuple[float, float]:
    """Computes (mean skill fit, mean rank of ranked) from assignments."""
    asgs = solution.assignments
    if not asgs:
        return 0.0, 0.0
    mean_fit = sum(a.skill_fit_score for a in asgs) / len(asgs)
    ranks = [a.preference_rank for a in asgs if a.preference_rank is not None]
    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
    return mean_fit, mean_rank


def _rank_key(rank: int | None) -> float:
    """Orders ranks so that an unranked allocation is worse than any rank."""
    return float("inf") if rank is None else float(rank)


def compare_two_solutions(
    solution_a: models.Solution,
    solution_b: models.Solution,
) -> dict[str, Any]:
    """Performs a trade-off comparison between two alternative solutions.

    All deltas are computed as (B - A). A positive mean-rank delta means
    students are on average worse off in B; a positive skill-fit delta means
    B achieves better technical alignment.

    Args:
        solution_a: Baseline solution.
        solution_b: Alternative solution to compare against A.

    Returns:
        Dictionary compatible with models.RunComparison.from_dict.
    """
    map_a = {a.student_id: a for a in solution_a.assignments}
    map_b = {b.student_id: b for b in solution_b.assignments}

    changed: list[dict[str, Any]] = []
    improved = 0
    worsened = 0
    for s_id in sorted(set(map_a) | set(map_b)):
        asg_a = map_a.get(s_id)
        asg_b = map_b.get(s_id)
        proj_a = asg_a.project_id if asg_a else None
        proj_b = asg_b.project_id if asg_b else None
        if proj_a == proj_b:
            continue

        rank_a = asg_a.preference_rank if asg_a else None
        rank_b = asg_b.preference_rank if asg_b else None
        if _rank_key(rank_b) < _rank_key(rank_a):
            rank_change = "IMPROVED"
            improved += 1
        elif _rank_key(rank_b) > _rank_key(rank_a):
            rank_change = "WORSENED"
            worsened += 1
        else:
            rank_change = "SAME"

        rank_delta = (
            rank_b - rank_a
            if rank_a is not None and rank_b is not None
            else None
        )
        changed.append(
            {
                "student_id": s_id,
                "student_name": s_id,
                "project_a": proj_a,
                "project_a_id": proj_a,
                "project_a_title": proj_a,
                "rank_a": rank_a,
                "skill_fit_a": asg_a.skill_fit_score if asg_a else None,
                "project_b": proj_b,
                "project_b_id": proj_b,
                "project_b_title": proj_b,
                "rank_b": rank_b,
                "skill_fit_b": asg_b.skill_fit_score if asg_b else None,
                "rank_delta": rank_delta,
                "rank_change": rank_change,
            }
        )

    # Aggregate indicators, falling back to assignment-level means.
    fb_fit_a, fb_rank_a = _fallback_means(solution_a)
    fb_fit_b, fb_rank_b = _fallback_means(solution_b)

    def pick(sol, section, key, fallback):
        value = _section_value(sol, section, key)
        return fallback if value is None else value

    fit_a = pick(solution_a, "efficiency", "mean_skill_fit", fb_fit_a)
    fit_b = pick(solution_b, "efficiency", "mean_skill_fit", fb_fit_b)
    rank_a_mean = pick(solution_a, "fairness", "mean_rank", fb_rank_a)
    rank_b_mean = pick(solution_b, "fairness", "mean_rank", fb_rank_b)
    gini_a = pick(solution_a, "fairness", "rank_gini", 0.0)
    gini_b = pick(solution_b, "fairness", "rank_gini", 0.0)
    worst_a = pick(solution_a, "fairness", "worst_rank", 0.0)
    worst_b = pick(solution_b, "fairness", "worst_rank", 0.0)
    unranked_a = pick(solution_a, "fairness", "unranked_count", 0.0)
    unranked_b = pick(solution_b, "fairness", "unranked_count", 0.0)

    delta_fit = round(fit_b - fit_a, 4)
    delta_rank = round(rank_b_mean - rank_a_mean, 4)
    delta_gini = round(gini_b - gini_a, 4)
    delta_worst = int(worst_b - worst_a)
    delta_unranked = int(unranked_b - unranked_a)

    # Human-readable trade-off statements (calculated facts only).
    summary = [
        f"{len(changed)} of {len(map_a)} student(s) receive a different "
        "project in the alternative.",
        f"{improved} student(s) improved and {worsened} worsened in "
        "preference rank.",
    ]
    if delta_fit > 0:
        summary.append(f"Mean skill fit increases by {delta_fit:+.4f}.")
    elif delta_fit < 0:
        summary.append(f"Mean skill fit decreases by {delta_fit:+.4f}.")
    if delta_rank > 0:
        summary.append(
            f"Mean preference rank worsens by {delta_rank:+.2f} positions."
        )
    elif delta_rank < 0:
        summary.append(
            f"Mean preference rank improves by {delta_rank:+.2f} positions."
        )
    if delta_gini != 0:
        summary.append(f"Rank inequality (Gini) changes by {delta_gini:+.4f}.")
    if not changed:
        summary.append("The two solutions are identical allocations.")

    return {
        "solution_a_id": solution_a.id,
        "solution_a_alt": solution_a.alternative_number,
        "solution_b_id": solution_b.id,
        "solution_b_alt": solution_b.alternative_number,
        "is_distinct": bool(changed),
        "changed_students_count": len(changed),
        "total_students": len(set(map_a) | set(map_b)),
        "changed_assignments": changed,
        "rank_improved_count": improved,
        "rank_worsened_count": worsened,
        "delta_mean_skill_fit": delta_fit,
        "delta_mean_rank": delta_rank,
        "delta_unranked_count": delta_unranked,
        "delta_metrics": {
            "mean_skill_fit_delta": delta_fit,
            "mean_rank_delta": delta_rank,
            "gini_delta": delta_gini,
            "worst_rank_delta": delta_worst,
            "unranked_count_delta": delta_unranked,
        },
        "trade_off_summary": summary,
    }