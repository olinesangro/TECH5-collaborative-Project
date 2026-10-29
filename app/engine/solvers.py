"""Mathematical optimization solvers and Strategy Pattern skeleton for TECH5.

In compliance with Section 5 and Section 8 of the TECH5 specification,
Team B is responsible for proposing and implementing an approved assignment
method (e.g. Integer Linear Programming, heuristic search, or stable matching).

This module defines the abstract strategy base class and the AssignmentEngine
that implements AssignmentEngineContract.

Method: binary Integer Linear Program solved with PuLP (CBC, or HiGHS where
the bundled CBC cannot run). Every strategy
shares the same hard constraints (exact allocation, capacity bounds, fixed
locks, mandatory-skill coverage, no-good cuts) and differs only in the
objective weights returned by ``calculate_cost_or_utility``. Ties in the
primary objective are broken lexicographically by ``secondary_utility``.
"""

import abc
import datetime
import logging
import os
import time
from typing import Any
import uuid

import pulp

from app import config
from app.contract import contract
from app.domain import models
from app.domain import validation
from app.engine import metrics
from app.persistence import repositories


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Solver backend selection
# -----------------------------------------------------------------------------
# PuLP ships its own CBC executable, but the macOS copy is built for Intel
# only, so it cannot start on Apple-silicon Macs without Rosetta ("Bad CPU
# type in executable"). The engine therefore probes CBC once and falls back to
# HiGHS (pip package ``highspy``, native on Apple silicon) only if CBC cannot
# run. Set TECH5_SOLVER=CBC or TECH5_SOLVER=HIGHS to force a backend.

_SOLVER_ENV_VAR = "TECH5_SOLVER"
_selected_backend: str | None = None


def _cbc_runs() -> bool:
    """Returns True if PuLP's bundled CBC executable can solve a tiny model."""
    probe = pulp.LpProblem("cbc_probe", pulp.LpMaximize)
    var = pulp.LpVariable("probe_x", cat="Binary")
    probe += var
    try:
        probe.solve(pulp.PULP_CBC_CMD(msg=False))
    except (OSError, pulp.PulpSolverError) as exc:
        logger.warning("Bundled CBC cannot run on this machine: %s", exc)
        return False
    return pulp.LpStatus[probe.status] == "Optimal"


def _highs_available() -> bool:
    """Returns True if the HiGHS solver (highspy) is installed."""
    return bool(pulp.HiGHS(msg=False).available())


def selected_backend() -> str:
    """Chooses the solver backend once per process and remembers it.

    Returns:
        "CBC" or "HIGHS".

    Raises:
        RuntimeError: If no usable solver is installed, with install advice.
    """
    global _selected_backend
    if _selected_backend is not None:
        return _selected_backend

    forced = os.environ.get(_SOLVER_ENV_VAR, "").strip().upper()
    if forced in ("CBC", "HIGHS"):
        _selected_backend = forced
    elif _cbc_runs():
        _selected_backend = "CBC"
    elif _highs_available():
        _selected_backend = "HIGHS"
    else:
        raise RuntimeError(
            "No usable MILP solver: PuLP's bundled CBC cannot run on this "
            "machine (on Apple-silicon Macs it is an Intel-only program) and "
            "HiGHS is not installed. Run: pip install -r requirements.txt "
            "(or pip install highspy==1.13)."
        )
    logger.info("Assignment engine using solver backend %s", _selected_backend)
    return _selected_backend


def _make_solver(time_limit_seconds: int | None) -> pulp.LpSolver:
    """Builds a silent solver instance for the selected backend."""
    if selected_backend() == "HIGHS":
        return pulp.HiGHS(msg=False, timeLimit=time_limit_seconds)
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)


# Relative tolerance used when fixing the primary objective before the
# lexicographic tie-break solve.
_LEXICOGRAPHIC_TOLERANCE = 1e-6


def _preference_utility(
    student: models.Student, project: models.Project, max_preferences: int
) -> float:
    """Borda-style preference score: |R|+1-r for ranked projects, else 0."""
    rank = student.get_preference_rank(project.id)
    if rank is None:
        return 0.0
    return float(max(max_preferences - rank + 1, 0))


def _skill_utility(
    student: models.Student, project: models.Project, max_preferences: int
) -> float:
    """Normalized skill fit sigma_{s,p} in [0, 1]."""
    score, _unmet = metrics.calculate_assignment_skill_fit(student, project)
    return score


class AssignmentStrategy(abc.ABC):
    """Abstract Base Class for student-to-project allocation strategies.

    Team B should derive concrete strategy subclasses implementing specific
    objective functions (e.g. preference satisfaction, skill fit, balanced).
    """

    def __init__(self, name: str, description: str) -> None:
        """Initializes the strategy with descriptive metadata.

        Args:
            name: Machine-readable identifier for the strategy.
            description: Human-readable explanation of the objective function.
        """
        self.name = name
        self.description = description

    @abc.abstractmethod
    def calculate_cost_or_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float:
        """Computes objective utility or cost weight for assigning student to project.

        Args:
            student: The candidate student.
            project: The target semester project.
            max_preferences: Maximum number of preferences ranked in cohort.

        Returns:
            Numerical weight reflecting desirability in optimization objective.
        """

    def secondary_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float | None:
        """Optional tie-break weight used after the primary objective.

        Among all allocations that are optimal for the primary objective, the
        solver picks the one maximizing this secondary weight. Returning None
        disables the tie-break stage.

        Args:
            student: The candidate student.
            project: The target semester project.
            max_preferences: Maximum number of preferences ranked in cohort.

        Returns:
            The secondary weight, or None if the strategy has no tie-break.
        """
        return None

    def solve(
        self,
        students: list[models.Student],
        projects: list[models.Project],
        fixed_assignments: list[models.FixedAssignment] | None = None,
        excluded_solutions: list[list[tuple[str, str]]] | None = None,
        time_limit_seconds: int | None = None,
    ) -> tuple[str, list[models.Assignment], float | None, dict[str, Any]]:
        """Solves the assignment problem under this strategy's formulation.

        Builds and solves a binary-assignment ILP whose objective weights come
        from `calculate_cost_or_utility`. Shared across strategies so each
        subclass only needs to define how a (student, project) pair is scored.

        Hard constraints (identical for every strategy):
        - Each student receives exactly one project.
        - Project headcount stays within [min_capacity, max_capacity].
        - Fixed assignments are locked (x = 1).
        - Mandatory skills: a staffed project has at least one assigned
          student meeting each of its mandatory requirements.
        - No-good cuts exclude every previously returned allocation, which
          guarantees each new solution is distinct.

        Args:
            students: List of students in the cohort.
            projects: List of available semester projects.
            fixed_assignments: Pre-assigned student-project pairs that must hold.
            excluded_solutions: Previous solution allocations to cut out.
            time_limit_seconds: Optional solver time limit per solve.

        Returns:
            A tuple of (solver_status, assignments, objective_value, diagnostics).
        """
        fixed_assignments = fixed_assignments or []
        excluded_solutions = excluded_solutions or []

        max_preferences = max(
            (len(s.preferences) for s in students),
            default=0,
        )

        prob = pulp.LpProblem(f"assignment_{self.name}", pulp.LpMaximize)

        # One binary decision variable per (student, project) pair.
        x: dict[tuple[str, str], pulp.LpVariable] = {
            (s.id, p.id): pulp.LpVariable(f"x_{s.id}_{p.id}", cat="Binary")
            for s in students
            for p in projects
        }

        # Objective: maximize total utility from calculate_cost_or_utility.
        primary_expr = pulp.lpSum(
            self.calculate_cost_or_utility(s, p, max_preferences) * x[s.id, p.id]
            for s in students
            for p in projects
        )
        prob += primary_expr

        # Each student must be assigned to exactly one project.
        for s in students:
            prob += (
                pulp.lpSum(x[s.id, p.id] for p in projects) == 1,
                f"one_project_{s.id}",
            )

        # Each project must stay within [min_capacity, max_capacity].
        for p in projects:
            total = pulp.lpSum(x[s.id, p.id] for s in students)
            prob += total >= p.min_capacity, f"min_cap_{p.id}"
            prob += total <= p.max_capacity, f"max_cap_{p.id}"

        # Mandatory skill coverage: u_p * sum_{s in Q_pk} x_sp >= sum_s x_sp.
        # If anyone is assigned to p, at least one qualified student must be.
        for p in projects:
            for req in p.requirements:
                if not req.is_mandatory:
                    continue
                qualified = [
                    s
                    for s in students
                    if s.get_skill_level(req.skill_id) >= req.minimum_level
                ]
                prob += (
                    p.max_capacity
                    * pulp.lpSum(x[s.id, p.id] for s in qualified)
                    >= pulp.lpSum(x[s.id, p.id] for s in students),
                    f"mandatory_{p.id}_{req.skill_id}",
                )

        # Pre-allocated assignments are locked in place.
        for fa in fixed_assignments:
            key = (fa.student_id, fa.project_id)
            if key not in x:
                raise ValueError(
                    f"Fixed assignment references unknown student/project pair: {key}"
                )
            prob += x[key] == 1, f"fixed_{fa.student_id}"

        # No-good cuts: forbid repeating any previously returned solution.
        for i, solution in enumerate(excluded_solutions):
            valid_pairs = [x[pair] for pair in solution if pair in x]
            if valid_pairs:
                prob += (
                    pulp.lpSum(valid_pairs) <= len(valid_pairs) - 1,
                    f"no_good_{i}",
                )

        solver = _make_solver(time_limit_seconds)
        started = time.perf_counter()
        solver_status = pulp.LpStatus[prob.solve(solver)]
        primary_value = (
            pulp.value(prob.objective) if solver_status == "Optimal" else None
        )

        # Lexicographic tie-break: keep the primary optimum, improve secondary.
        tie_break_used = False
        if solver_status == "Optimal" and primary_value is not None:
            secondary_weights = {
                (s.id, p.id): self.secondary_utility(s, p, max_preferences)
                for s in students
                for p in projects
            }
            if all(w is not None for w in secondary_weights.values()):
                tolerance = _LEXICOGRAPHIC_TOLERANCE * max(1.0, abs(primary_value))
                prob += (
                    primary_expr >= primary_value - tolerance,
                    "lexicographic_primary",
                )
                prob.setObjective(
                    pulp.lpSum(
                        secondary_weights[key] * var for key, var in x.items()
                    )
                )
                second_status = pulp.LpStatus[prob.solve(solver)]
                if second_status == "Optimal":
                    tie_break_used = True
                    primary_value = pulp.value(primary_expr)
                else:
                    solver_status = second_status
        solve_seconds = round(time.perf_counter() - started, 4)

        fixed_pairs = {(fa.student_id, fa.project_id) for fa in fixed_assignments}

        assignments: list[models.Assignment] = []
        if solver_status == "Optimal":
            for s in students:
                for p in projects:
                    var = x[s.id, p.id]
                    if var.value() is not None and var.value() > 0.5:
                        skill_fit_score, unmet = metrics.calculate_assignment_skill_fit(
                            s, p
                        )
                        assignments.append(
                            models.Assignment(
                                student_id=s.id,
                                project_id=p.id,
                                is_fixed=(s.id, p.id) in fixed_pairs,
                                preference_rank=s.get_preference_rank(p.id),
                                skill_fit_score=skill_fit_score,
                                unmet_requirements=unmet,
                            )
                        )
                        break

        objective_value = (
            round(primary_value, 6)
            if solver_status == "Optimal" and primary_value is not None
            else None
        )

        diagnostics = {
            "strategy": self.name,
            "solver_status": solver_status,
            "num_students": len(students),
            "num_projects": len(projects),
            "num_variables": len(x),
            "num_constraints": len(prob.constraints),
            "num_fixed": len(fixed_assignments),
            "num_excluded_cuts": len(excluded_solutions),
            "solver_backend": selected_backend(),
            "lexicographic_tie_break": tie_break_used,
            "solve_time_seconds": solve_seconds,
        }

        return solver_status, assignments, objective_value, diagnostics


class PreferenceOrientedSolver(AssignmentStrategy):
    """Maximizes aggregate student preference satisfaction (Section 7.1).

    Primary: Borda score sum_s sum_p (|R|+1-r_sp) x_sp.
    Tie-break: total skill fit among preference-optimal allocations.
    """

    def __init__(self) -> None:
        super().__init__(
            name="PREFERENCE",
            description=(
                "Prioritizes assignments matching each student's "
                "highest-ranked project."
            ),
        )

    def calculate_cost_or_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float:
        return _preference_utility(student, project, max_preferences)

    def secondary_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float | None:
        return _skill_utility(student, project, max_preferences)


class SkillFitOrientedSolver(AssignmentStrategy):
    """Maximizes aggregate student-project technical skill fit (Section 7.2).

    Primary: total skill fit sum_s sum_p sigma_sp x_sp.
    Tie-break: Borda preference score among skill-optimal allocations.
    """

    def __init__(self) -> None:
        super().__init__(
            name="SKILL_FIT",
            description=(
                "Prioritizes assignments where students best satisfy "
                "project skill requirements."
            ),
        )

    def calculate_cost_or_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float:
        # Scale to a comparable magnitude with preference-rank utilities.
        return _skill_utility(student, project, max_preferences) * max(
            max_preferences, 1
        )

    def secondary_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float | None:
        return _preference_utility(student, project, max_preferences)


class BalancedTradeoffSolver(AssignmentStrategy):
    """Blends preference satisfaction and skill fit into one objective.

    Primary: sum_s sum_p [w * pi_sp + (1 - w) * sigma_sp] x_sp, where pi_sp is
    the Borda score normalized to [0, 1]. No tie-break stage.
    """

    def __init__(self, preference_weight: float = 0.5) -> None:
        super().__init__(
            name="BALANCED",
            description=(
                "Balances preference satisfaction against technical skill fit."
            ),
        )
        self.preference_weight = preference_weight

    def calculate_cost_or_utility(
        self,
        student: models.Student,
        project: models.Project,
        max_preferences: int,
    ) -> float:
        pref_component = _preference_utility(
            student, project, max_preferences
        ) / max(max_preferences, 1)
        skill_component = _skill_utility(student, project, max_preferences)
        return (
            self.preference_weight * pref_component
            + (1.0 - self.preference_weight) * skill_component
        )


# Strategies are resolved lazily by key rather than stored on the engine
# instance, so AssignmentEngine.strategies can stay empty at construction
# (test_solvers.py::test_assignment_engine_initialization depends on this).
_STRATEGY_REGISTRY: dict[str, type[AssignmentStrategy]] = {
    "PREFERENCE": PreferenceOrientedSolver,
    "SKILL_FIT": SkillFitOrientedSolver,
    "BALANCED": BalancedTradeoffSolver,
}


def _resolve_strategy(strategy_key: str) -> AssignmentStrategy:
    """Instantiates the strategy for a given key, defaulting to PREFERENCE."""
    strategy_cls = _STRATEGY_REGISTRY.get(strategy_key, PreferenceOrientedSolver)
    return strategy_cls()


def _strategy_sequence(primary_key: str, count: int) -> list[AssignmentStrategy]:
    """Orders the strategies used for successive alternatives.

    Alternative 1 always uses the selected primary strategy. Alternatives 2
    and 3 use the remaining strategies so the user sees genuine trade-offs;
    any further alternatives return to the primary strategy.

    Args:
        primary_key: The strategy chosen by the user.
        count: Number of alternatives requested.

    Returns:
        Strategy instances, one per requested alternative.
    """
    primary = _resolve_strategy(primary_key)
    others = [key for key in _STRATEGY_REGISTRY if key != primary.name]
    sequence: list[AssignmentStrategy] = [primary]
    sequence += [_resolve_strategy(key) for key in others]
    while len(sequence) < count:
        sequence.append(_resolve_strategy(primary.name))
    return sequence[:count]


class AssignmentEngine(contract.AssignmentEngineContract):
    """Main Assignment Engine orchestrating validation, execution, and persistence.

    Implements the internal component contract (AssignmentEngineContract)
    connecting Component B to Component A.
    """

    def __init__(
        self, run_repo: repositories.RunRepository | None = None
    ) -> None:
        """Initializes the AssignmentEngine with persistence repositories.

        Args:
            run_repo: Optional repository for persisting runs and solutions.
        """
        self.run_repo = (
            run_repo if run_repo is not None else repositories.RunRepository()
        )
        # Intentionally left empty: strategies are resolved on demand by
        # _resolve_strategy() in execute_run, not stored here.
        self.strategies: dict[str, AssignmentStrategy] = {}

    def validate_inputs(
        self, request: contract.RunRequest
    ) -> contract.ValidationOutcome:
        """Validates cohort and constraints before attempting optimization.

        Args:
            request: The run request containing students, projects, etc.

        Returns:
            ValidationOutcome with validity, feasibility, errors, and warnings.
        """
        val_result = validation.validate_cohort(
            students=request.students,
            projects=request.projects,
            skills=request.skills,
            fixed_assignments=request.fixed_assignments,
        )
        return contract.ValidationOutcome(
            is_valid=val_result.is_valid,
            is_feasible=val_result.is_feasible,
            errors=list(val_result.errors),
            warnings=list(val_result.warnings),
            infeasibility_reasons=list(val_result.infeasibility_reasons),
        )

    def _stopped_run(
        self,
        run_id: str,
        created_at: str,
        status: contract.RunStatus,
        configuration: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> models.AssignmentRun:
        """Persists and returns a run that produced no solutions."""
        run = models.AssignmentRun(
            id=run_id,
            created_at=created_at,
            status=status.value,
            configuration=configuration,
            solutions=[],
            diagnostics={"distinct_alternatives": 0, **diagnostics},
        )
        self.run_repo.save_run(run)
        return run

    def execute_run(
        self, request: contract.RunRequest
    ) -> models.AssignmentRun:
        """Executes the assignment algorithm, calculates metrics, and persists run.

        In accordance with Section 8 of the specification:
        1. Validates cohort feasibility and input integrity.
        2. Synchronizes entities to maintain relational integrity.
        3. Runs the assignment solver to generate distinct alternatives,
           applying no-good cuts until max_alternatives is reached or no
           further distinct feasible solution exists.
        4. Calculates indicators for each alternative using app.engine.metrics.
        5. Persists run and solutions via self.run_repo.
        6. Returns the populated models.AssignmentRun.

        Args:
            request: The run request containing students, projects, etc.

        Returns:
            The created AssignmentRun populated with alternative solutions.
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        requested = max(1, request.max_alternatives)
        strategies = _strategy_sequence(request.strategy_key, requested)
        base_configuration = {
            "max_alternatives": request.max_alternatives,
            "primary_strategy": strategies[0].name,
            "strategy_sequence": [s.name for s in strategies],
            "engine": "AssignmentEngine",
            "method": "ILP (PuLP) with no-good cuts",
            "solver_backend": selected_backend(),
            "notes": request.notes,
        }

        # 1a. Structural infeasibility proven before solving.
        val_outcome = self.validate_inputs(request)
        if not val_outcome.is_feasible:
            return self._stopped_run(
                run_id,
                created_at,
                contract.RunStatus.INFEASIBLE,
                base_configuration,
                {
                    "solver_status": "INFEASIBLE",
                    "message": "Infeasibility proven by pre-solve validation.",
                    "reasons": val_outcome.infeasibility_reasons,
                    "errors": [
                        e
                        for e in val_outcome.errors
                        if e not in val_outcome.infeasibility_reasons
                    ],
                    "warnings": val_outcome.warnings,
                },
            )

        # 1b. Invalid references (unknown IDs, duplicates, conflicting locks)
        # are rejected before anything touches the database or solver.
        if not val_outcome.is_valid:
            return self._stopped_run(
                run_id,
                created_at,
                contract.RunStatus.INVALID_INPUT,
                base_configuration,
                {
                    "solver_status": "NOT_SOLVED",
                    "message": "Input rejected by validation; fix the "
                    "errors below and run again.",
                    "errors": val_outcome.errors,
                    "warnings": val_outcome.warnings,
                },
            )

        # 2. Synchronize entities to maintain relational integrity.
        if request.skills:
            skill_repo = repositories.SkillRepository()
            for sk in request.skills:
                skill_repo.save(sk)

        proj_repo = repositories.ProjectRepository()
        for proj in request.projects:
            proj_repo.save(proj)

        stud_repo = repositories.StudentRepository()
        for stud in request.students:
            stud_repo.save(stud)

        if request.fixed_assignments:
            fixed_repo = repositories.FixedAssignmentRepository()
            for fa in request.fixed_assignments:
                fixed_repo.save(fa)

        # 3-4. Generate distinct alternatives with no-good cuts.
        time_limit = config.Config.PULP_TIME_LIMIT_SECONDS
        solutions: list[models.Solution] = []
        excluded_solutions: list[list[tuple[str, str]]] = []
        last_status = "NOT_SOLVED"

        for alt_idx, strategy in enumerate(strategies):
            status, assignments, objective_value, solve_diag = strategy.solve(
                students=request.students,
                projects=request.projects,
                fixed_assignments=request.fixed_assignments,
                excluded_solutions=excluded_solutions,
                time_limit_seconds=time_limit,
            )
            last_status = status
            # The cuts are identical whichever objective is used, so an
            # infeasible model here means no further distinct allocation
            # exists for any strategy.
            if status != "Optimal" or not assignments:
                break

            indicators = metrics.compute_solution_indicators(
                students=request.students,
                projects=request.projects,
                assignments=assignments,
                fixed_assignments=request.fixed_assignments,
            )

            solution = models.Solution(
                id=str(uuid.uuid4()),
                run_id=run_id,
                alternative_number=alt_idx + 1,
                strategy_name=strategy.name,
                objective_value=objective_value,
                assignments=assignments,
                indicators=indicators,
                warnings=list(indicators["feasibility"]["violations"]),
                diagnostics=solve_diag,
            )
            solutions.append(solution)
            excluded_solutions.append(
                [(a.student_id, a.project_id) for a in assignments]
            )

        run_diagnostics: dict[str, Any] = {
            "solver_status": last_status,
            "distinct_alternatives": len(solutions),
            "requested_alternatives": requested,
            "warnings": val_outcome.warnings,
        }

        if not solutions:
            run_diagnostics["message"] = (
                "The solver proved that no allocation satisfies all hard "
                "constraints together, although each passed the pre-solve "
                "checks individually."
            )
            run_diagnostics["reasons"] = [
                f"Solver status '{last_status}' for the model with exact "
                "allocation, capacity bounds, fixed locks and mandatory "
                "skill coverage.",
                "Typical causes: mandatory skills held by too few students "
                "for the projects that must be staffed (min capacity > 0), "
                "or fixed locks that leave no room for qualified students.",
            ]
            status_value = contract.RunStatus.INFEASIBLE.value
        else:
            status_value = contract.RunStatus.COMPLETED.value
            if len(solutions) < requested:
                run_diagnostics["message"] = (
                    f"Only {len(solutions)} distinct feasible allocation(s) "
                    f"exist; {requested} were requested. No duplicates were "
                    "generated."
                )
            else:
                run_diagnostics["message"] = (
                    f"{len(solutions)} distinct alternatives generated "
                    f"({', '.join(s.strategy_name for s in solutions)})."
                )

        run = models.AssignmentRun(
            id=run_id,
            created_at=created_at,
            status=status_value,
            configuration=base_configuration,
            solutions=solutions,
            diagnostics=run_diagnostics,
        )
        self.run_repo.save_run(run)
        return run

    def run_assignment(
        self,
        students: list[models.Student],
        projects: list[models.Project],
        skills: list[models.Skill] | None = None,
        fixed_assignments: list[models.FixedAssignment] | None = None,
        strategy_name: str = "PREFERENCE",
        max_alternatives: int = 3,
        notes: str = "",
    ) -> models.AssignmentRun:
        """Convenience method delegating to execute_run via RunRequest.

        Args:
            students: Student profiles.
            projects: Project specifications.
            skills: Optional skills catalog.
            fixed_assignments: Optional pre-allocated locks.
            strategy_name: Strategy key name.
            max_alternatives: Max solutions to generate.
            notes: Administrative description or notes for the run.

        Returns:
            The created AssignmentRun.
        """
        req = contract.RunRequest(
            students=students,
            projects=projects,
            skills=skills or [],
            fixed_assignments=fixed_assignments or [],
            strategy_key=strategy_name,
            max_alternatives=max_alternatives,
            notes=notes,
        )
        return self.execute_run(req)

    def list_runs(self, limit: int = 20) -> list[models.AssignmentRun]:
        """Retrieves historical assignment runs from repository.

        Args:
            limit: Maximum number of runs to retrieve.

        Returns:
            List of historical AssignmentRun domain objects.
        """
        return self.run_repo.list_runs(limit=limit)

    def get_run(self, run_id: str) -> models.AssignmentRun | None:
        """Retrieves a specific run by identifier.

        Args:
            run_id: Unique run identifier.

        Returns:
            The AssignmentRun object, or None if not found.
        """
        return self.run_repo.get_run(run_id)

    def get_solution(self, solution_id: str) -> models.Solution | None:
        """Retrieves a specific solution and its student allocations.

        Args:
            solution_id: Unique solution identifier.

        Returns:
            The Solution object with assignments, or None if not found.
        """
        return self.run_repo.get_solution(solution_id)

    def compare_solutions(
        self, solution_a_id: str, solution_b_id: str
    ) -> models.RunComparison:
        """Compares two alternative solutions side-by-side using metrics module.

        Args:
            solution_a_id: Identifier of the baseline solution.
            solution_b_id: Identifier of the comparison solution.

        Returns:
            RunComparison containing delta metrics and changed assignments.

        Raises:
            KeyError: If either solution cannot be found.
        """
        sol_a = self.run_repo.get_solution(solution_a_id)
        sol_b = self.run_repo.get_solution(solution_b_id)
        if sol_a is None or sol_b is None:
            raise KeyError(
                f"Solutions not found: {solution_a_id}, {solution_b_id}"
            )
        comp_dict = metrics.compare_two_solutions(sol_a, sol_b)
        return models.RunComparison.from_dict(comp_dict)