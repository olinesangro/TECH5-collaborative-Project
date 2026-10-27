"""Tests for Component A (Visualization & Recommendation) web workflows."""

import pathlib

import pytest

from app import create_app
from app.domain import models
from app.persistence import repositories


def test_dashboard_loads(tmp_path: pathlib.Path) -> None:
    """Verifies dashboard overview page rendering."""
    db_file = tmp_path / "test_ui_dash.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    dash_res = client.get("/")
    assert dash_res.status_code == 200
    assert b"TECH5" in dash_res.data


def test_role_switcher(tmp_path: pathlib.Path) -> None:
    """Verifies role switching in session."""
    db_file = tmp_path / "test_role.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    res = client.post(
        "/set-role", data={"role": "STUDENT"}, follow_redirects=True
    )
    assert res.status_code == 200
    assert b"Switched role to Student Viewer" in res.data


def test_student_and_fixed_assignment_workflow(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies student and fixed-assignment creation workflows."""
    db_file = tmp_path / "test_stud_fixed.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    # Seed project P1 so student preference foreign key is valid
    repositories.ProjectRepository().save(
        models.Project(
            id="P1",
            title="FinTech Risk",
            min_capacity=1,
            max_capacity=2,
        )
    )

    # Create student
    res_stud = client.post(
        "/students/new",
        data={
            "id": "S1",
            "display_name": "Alice Chen",
            "preferences_text": "P1",
        },
        follow_redirects=True,
    )
    assert res_stud.status_code == 200
    assert b"Alice Chen" in res_stud.data

    # Create fixed assignment
    res_fixed = client.post(
        "/fixed-assignments/new",
        data={
            "student_id": "S1",
            "project_id": "P1",
            "status": "APPROVED",
            "reason": "Pre-matched",
        },
        follow_redirects=True,
    )
    assert res_fixed.status_code == 200
    assert b"Pre-matched" in res_fixed.data


def test_scenario_loading_and_optimization_trigger(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies benchmark scenario loading, execution, and result views."""
    db_file = tmp_path / "test_scenario_run.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    # Load standard benchmark
    load_res = client.post("/load-scenario/standard", follow_redirects=True)
    assert load_res.status_code == 200
    assert b"Standard Cohort" in load_res.data

    # Trigger optimization run (falls back to mock engine if solver unimplemented)
    run_res = client.post(
        "/runs/trigger",
        data={
            "primary_strategy": "PREFERENCE",
            "max_alternatives": 3,
        },
        follow_redirects=True,
    )
    assert run_res.status_code == 200
    assert b"Optimization run completed" in run_res.data

    runs = repositories.RunRepository().list_runs(limit=1)
    assert len(runs) == 1
    run = runs[0]
    assert len(run.solutions) >= 1

    # View solution detail
    sol_id = run.solutions[0].id
    sol_res = client.get(f"/solutions/{sol_id}")
    assert sol_res.status_code == 200
    assert b"Student-to-Project Allocations" in sol_res.data

    # View recommendations
    recs_res = client.get(f"/recommendations/{run.id}")
    assert recs_res.status_code == 200
    assert b"Transparent Decision Recommendations" in recs_res.data
    assert b"Calculated Evidence" in recs_res.data
    assert b"Policy Advice" in recs_res.data


def test_create_student_preference_validation(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies preference validation when creating a student."""
    db_file = tmp_path / "test_stud_pref_val.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    repositories.ProjectRepository().save(
        models.Project(
            id="P1",
            title="Project One",
            min_capacity=1,
            max_capacity=2,
        )
    )

    # 1. Invalid project preference should flash error and not crash
    res_invalid = client.post(
        "/students/new",
        data={
            "id": "S10",
            "display_name": "Test Student",
            "preferences_text": "P99, P100",
        },
        follow_redirects=True,
    )
    assert res_invalid.status_code == 200
    assert b"Invalid project ID(s) in preferences" in res_invalid.data

    # 2. Case-insensitive project preference should normalize and succeed
    res_valid = client.post(
        "/students/new",
        data={
            "id": "S10",
            "display_name": "Test Student",
            "preferences_text": "p1",
        },
        follow_redirects=True,
    )
    assert res_valid.status_code == 200
    assert b"created successfully" in res_valid.data

    saved = repositories.StudentRepository().get_by_id("S10")
    assert saved is not None
    assert len(saved.preferences) == 1
    assert saved.preferences[0].project_id == "P1"


def test_skills_crud_workflow(tmp_path: pathlib.Path) -> None:
    """Verifies skill creation and deletion (Team A implementation)."""
    db_file = tmp_path / "test_skill.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    response = client.post(
        "/skills/new",
        data={
            "id": "py",
            "name": "Python",
            "description": "Python Programming",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Python" in response.data

    response = client.post("/skills/py/delete", follow_redirects=True)
    assert response.status_code == 200
    assert b"Python Programming" not in response.data


def test_projects_crud_workflow(tmp_path: pathlib.Path) -> None:
    """Verifies project create, edit and delete (Team A implementation)."""
    db_file = tmp_path / "test_proj.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    client.post(
        "/skills/new",
        data={"id": "py", "name": "Python", "description": ""},
        follow_redirects=True,
    )

    response = client.post(
        "/projects/new",
        data={
            "id": "P1",
            "title": "FinTech Risk",
            "description": "Risk analytics",
            "min_capacity": 1,
            "max_capacity": 2,
            "req_level_py": 3,
            "req_mand_py": "1",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"FinTech Risk" in response.data

    with app.app_context():
        project = repositories.ProjectRepository().get_by_id("P1")
    assert project is not None
    assert project.max_capacity == 2
    assert len(project.requirements) == 1
    assert project.requirements[0].minimum_level == 3
    assert project.requirements[0].is_mandatory

    # The edit form shows the saved requirement, not a blank form.
    res = client.get("/projects/P1/edit")
    assert b'value="3"' in res.data

    response = client.post(
        "/projects/P1/edit",
        data={
            "title": "FinTech Risk v2",
            "min_capacity": 0,
            "max_capacity": 3,
            "req_level_py": 0,
        },
        follow_redirects=True,
    )
    assert b"FinTech Risk v2" in response.data
    with app.app_context():
        project = repositories.ProjectRepository().get_by_id("P1")
    assert project.max_capacity == 3
    assert project.requirements == []

    response = client.post(
        "/projects/new",
        data={
            "id": "P2",
            "title": "Bad capacity",
            "min_capacity": 5,
            "max_capacity": 2,
        },
        follow_redirects=True,
    )
    assert b"cannot exceed" in response.data

    response = client.post("/projects/P1/delete", follow_redirects=True)
    assert response.status_code == 200
    with app.app_context():
        assert repositories.ProjectRepository().get_by_id("P1") is None


def test_compare_solutions_view(tmp_path: pathlib.Path) -> None:
    """Verifies the side-by-side comparison page (Team A implementation)."""
    db_file = tmp_path / "test_comp.db"
    app = create_app({"TESTING": True, "DATABASE_PATH": db_file})
    client = app.test_client()

    # Unknown solutions must show a message instead of crashing.
    res = client.get("/compare?sol1=S1&sol2=S2")
    assert res.status_code == 200
    assert b"not found" in res.data

    client.post("/load-scenario/standard", follow_redirects=True)
    client.post(
        "/runs/trigger",
        data={"primary_strategy": "PREFERENCE", "max_alternatives": 3},
        follow_redirects=True,
    )
    run = repositories.RunRepository().list_runs(limit=1)[0]
    assert len(run.solutions) >= 2
    sol_a, sol_b = run.solutions[0], run.solutions[1]

    res = client.get(f"/compare?sol1={sol_a.id}&sol2={sol_b.id}")
    assert res.status_code == 200
    assert b"Detailed Allocation Shifts" in res.data
    assert b"Rank Gini" in res.data

    # Only sol1 given: another alternative from the same run is picked.
    res = client.get(f"/compare?sol1={sol_a.id}")
    assert res.status_code == 200
    assert b"Detailed Allocation Shifts" in res.data

    # Comparing a solution with itself is rejected with a warning.
    res = client.get(f"/compare?sol1={sol_a.id}&sol2={sol_a.id}")
    assert res.status_code == 200
    assert b"two different solutions" in res.data