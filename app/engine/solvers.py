"""Mathematical optimization solvers and Strategy Pattern skeleton for TECH5.

In compliance with Section 5 and Section 8 of the TECH5 specification,
Team B is responsible for proposing and implementing an approved assignment
method (e.g. Integer Linear Programming, heuristic search, or stable matching).

This module defines the abstract strategy base class and the AssignmentEngine
that implements AssignmentEngineContract.
"""

import abc
import datetime
import logging
from typing import Any
import uuid

import pulp

from app.contract import contract
from app.domain import models
from app.domain import validation
from app.engine import metrics
from app.persistence import repositories


logger = logging.getLogger(__name__)


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

    def solve(
        self,
        students: list[models.Student],
        projects: list[models.Project],
        fixed_assignments: list[models.FixedAssignment] | None = None,
        excluded_solutions: list[list[tuple[str, str]]] | None = None,
    ) -> tuple[str, list[models.Assignment], float | None, dict[str, Any]]:
        """Solves the assignment problem under this strategy's formulation.

        Builds and solves a binary-assignment ILP whose objective weights come
        from `calculate_cost_or_utility`. Shared across strategies so each
        subclass only needs to define how a (student, project) pair is scored.

        In accordance with Section 8 of the specification:
        - Assigns students to projects respecting capacities [min_capacity, max_capacity].
        - Enforces each student receives exactly one project allocation.
        - Strictly preserves pre-allocated fixed assignments.
        - Supports integer no-good cuts (excluded_solutions) to guarantee distinct
          Pareto alternatives.

        Args:
            students: List of students in the cohort.
            projects: List of available semester projects.
            fixed_assignments: Pre-assigned student-project pairs that must hold.
            excluded_solutions: Previous solution allocations to cut out.

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
        prob += pulp.lpSum(
            self.calculate_cost_or_utility(s, p, max_preferences) * x[s.id, p.id]
            for s in students
            for p in projects
        )

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

        solver_status_code = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        solver_status = pulp.LpStatus[solver_status_code]

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
            pulp.value(prob.objective) if solver_status == "Optimal" else None
        )

        diagnostics = {
            "strategy": self.name,
            "solver_status": solver_status,
            "num_students": len(students),
            "num_projects": len(projects),
            "num_fixed": len(fixed_assignments),
            "num_excluded_cuts": len(excluded_solutions),
        }

        return solver_status, assignments, objective_value, diagnostics


class PreferenceOrientedSolver(AssignmentStrategy):
    """Maximizes aggregate student preference satisfaction (Section 7.1)."""

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
        rank = student.get_preference_rank(project.id)
        if rank is None:
            return 0.0
        return float(max_preferences - rank + 1)


class SkillFitOrientedSolver(AssignmentStrategy):
    """Maximizes aggregate student-project technical skill fit (Section 7.2)."""

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
        score, _unmet = metrics.calculate_assignment_skill_fit(student, project)
        # Scale to a comparable magnitude with preference-rank utilities.
        return score * max(max_preferences, 1)


class BalancedTradeoffSolver(AssignmentStrategy):
    """Blends preference satisfaction and skill fit into one objective."""

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
        rank = student.get_preference_rank(project.id)
        pref_component = (
            0.0
            if rank is None
            else (max_preferences - rank + 1) / max(max_preferences, 1)
        )
        skill_component, _unmet = metrics.calculate_assignment_skill_fit(
            student, project
        )
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

    def execute_run(
        self, request: contract.RunRequest
    ) -> models.AssignmentRun:
        """Executes the assignment algorithm, calculates metrics, and persists run.

        In accordance with Section 8 of the specification:
        1. Validates cohort feasibility.
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
        base_configuration = {
            "max_alternatives": request.max_alternatives,
            "primary_strategy": request.strategy_key,
            "engine": "AssignmentEngine",
        }

        val_outcome = self.validate_inputs(request)
        if not val_outcome.is_feasible:
            run = models.AssignmentRun(
                id=run_id,
                created_at=created_at,
                status=contract.RunStatus.INFEASIBLE.value,
                configuration=base_configuration,
                solutions=[],
                diagnostics={
                    "solver_status": "INFEASIBLE",
                    "reasons": val_outcome.infeasibility_reasons,
                    "distinct_alternatives": 0,
                },
            )
            self.run_repo.save_run(run)
            return run

        # Synchronize entities to maintain relational integrity.
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

        strategy = _resolve_strategy(request.strategy_key)

        solutions: list[models.Solution] = []
        excluded_solutions: list[list[tuple[str, str]]] = []
        last_status = "NOT_SOLVED"

        for alt_idx in range(max(1, request.max_alternatives)):
            status, assignments, objective_value, solve_diag = strategy.solve(
                students=request.students,
                projects=request.projects,
                fixed_assignments=request.fixed_assignments,
                excluded_solutions=excluded_solutions,
            )
            last_status = status
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
                warnings=[],
                diagnostics=solve_diag,
            )
            solutions.append(solution)
            excluded_solutions.append(
                [(a.student_id, a.project_id) for a in assignments]
            )

        status_value = (
            contract.RunStatus.COMPLETED.value
            if solutions
            else contract.RunStatus.INFEASIBLE.value
        )

        run = models.AssignmentRun(
            id=run_id,
            created_at=created_at,
            status=status_value,
            configuration=base_configuration,
            solutions=solutions,
            diagnostics={
                "solver_status": last_status,
                "distinct_alternatives": len(solutions),
            },
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