# TECH5 Assignment Support System: Reproducible Demonstration Walkthrough

## 1. Demonstration Overview

This walkthrough reproduces every core capability of the
**Student-to-Project Assignment Decision Support System** on the four
benchmark cohorts built into the application. All numbers below were produced
by our implementation (ILP solved with PuLP/CBC) and are deterministic for the
benchmark data.

### Setup and Quick Start (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest -v
python run.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

- `pytest -v` should end with **38 passed, 1 skipped, 12 xpassed** and no
  failures. The skipped test is the starter-carcass check
  `test_assignment_engine_raises_not_implemented`, which the course
  instructions say to ignore.
- Open **http://127.0.0.1:5000** in a browser. Stop the server with Ctrl+C.

---

## 2. Scenario 1: Standard Cohort (Trade-Off Benchmark)

### 2.1 Objective

Shows distinct alternatives generated with no-good cuts, and the trade-off
between student preference satisfaction and project skill fit.

### 2.2 Cohort

- **Students:** 10 (`S01`–`S10`) with five skills.
- **Projects:** 4 (`P1`–`P4`), total maximum capacity 12 for 10 students.
- **Fixed assignments:** none.

### 2.3 Steps

1. Click **Scenarios**, then **Load Standard Benchmark →**.
2. On the **Dashboard**, keep the strategy *Student Preference Maximization*
   and *3 Distinct Solutions*, then click **Launch Optimization Run →**.
3. The run page reports *"Optimization run completed! Found 3 distinct
   alternative solutions."*

| Alternative | Strategy   | Mean skill fit | Mean rank | Rank Gini | Choices          |
|-------------|------------|----------------|-----------|-----------|------------------|
| 1           | PREFERENCE | 0.983          | 1.0       | 0.000     | 10 × 1st         |
| 2           | SKILL_FIT  | 1.000          | 1.1       | 0.082     | 9 × 1st, 1 × 2nd |
| 3           | BALANCED   | 0.983          | 1.1       | 0.082     | 9 × 1st, 1 × 2nd |

Alternatives 2 and 3 are the best allocations for their strategy *among
allocations that differ from the earlier ones* (no-good cuts). Alternative 3
is therefore no better than alternative 1 on these indicators; the
recommendations page evaluates alternatives by their indicators, not by
their strategy label.

### 2.4 Side-by-Side Comparison

1. Click **Compare**. Choose Solution A = *Alt #1 – PREFERENCE* and
   Solution B = *Alt #2 – SKILL_FIT*, then click **Compare Allocations →**.
2. Expected result: **1 student changed** (`S06`, from 1st to 2nd choice),
   Δ average skill fit **+1.7%**, Δ average rank **+0.1**,
   Δ rank Gini **+0.0818**.

### 2.5 Transparent Recommendations

1. Open the run (Dashboard → **View Solutions →**) and click
   **View Recommendations →**.
2. Three cards appear (Student Preference, Project Skill Quality, Balanced).
   Each shows the **criteria used**, the **calculated evidence** and the
   **policy advice** separately. When one alternative wins several
   priorities, or two alternatives tie, a **Decision notes** box says so.

### 2.6 Headless Contract Execution (Python)

The engine runs without the web server, through the component contract:

```python
from app import create_app
from app.contract import contract
from app.engine import solvers
from app.samples import demo_data

create_app({"DATABASE_PATH": "demo.db"})  # initialises the database

data = demo_data.get_standard_scenario()
request = contract.RunRequest(
    students=data["students"],
    projects=data["projects"],
    skills=data["skills"],
    strategy_key="PREFERENCE",
    max_alternatives=3,
)
run = solvers.AssignmentEngine().execute_run(request)
print(run.status, len(run.solutions))
for sol in run.solutions:
    print(sol.alternative_number, sol.strategy_name,
          sol.indicators["fairness"]["mean_rank"],
          sol.indicators["efficiency"]["mean_skill_fit"])
```

Expected output: `COMPLETED 3`, followed by the three rows of the table in
section 2.3.

---

## 3. Scenario 2: Fixed Assignment and Rerun

### 3.1 Objective

Shows that administrative locks ($x_{s^*,p^*} = 1$) are always preserved,
and how a lock affects other students.

### 3.2 Cohort

The standard cohort, with student **`S02`** locked to **`P2`** (Cloud Data
Pipeline & Lakehouse), although `P2` is only S02's 3rd choice.

### 3.3 Steps

1. Click **Scenarios**, then **Load Fixed Assignment Scenario →**.
2. Click **Fixed Locks**: `S02 → P2` is listed with status `APPROVED`.
3. On the **Dashboard**, click **Launch Optimization Run →**.
4. In every alternative, `S02` is in `P2` with a **🔒 FIXED LOCK** badge.
   Alternative 1 has mean rank **1.2** (9 students get their 1st choice,
   S02 gets its 3rd) and mean skill fit **0.967**.
5. Open a solution (**Explore Allocations →**) and click **🔒 Lock & Rerun**
   on any other student. The lock is saved and a new run starts at once; the
   new run keeps both locks.

---

## 4. Scenario 3: Capacity Deficit (Infeasibility)

### 4.1 Objective

Shows that structural infeasibility ($\sum_p u_p < |S|$) is detected before
solving and explained to the user.

### 4.2 Steps

1. Click **Scenarios**, then **Load Infeasible Capacity Scenario →**.
2. The Dashboard shows 10 students, **Total Max Capacity 6** and
   *⚠ Capacity Deficit*.
3. Click **Launch Optimization Run →**. Status **INFEASIBLE**, no solutions,
   and the diagnostics read:

   ```
   Insufficient overall capacity: total cohort size (10) exceeds aggregate maximum project capacity (6).
   ```

---

## 5. Scenario 4: Contradictory Fixed Assignments

### 5.1 Objective

Shows that locks exceeding a project's capacity ($|F_p| > u_p$) are rejected
with a clear reason.

### 5.2 Steps

1. Click **Scenarios**, then **Load Contradictory Scenario →**
   (4 students locked to `P1`, maximum capacity 3).
2. Click **Launch Optimization Run →**. Status **INFEASIBLE**, no solutions,
   and the diagnostics read:

   ```
   Fixed assignments (4) for project 'P1' exceed its maximum capacity (3).
   ```

---

## 6. Data Entry (Optional)

1. **Skills:** add a skill with the form on the right; delete it again.
2. **Projects:** **+ Add New Project**, set capacities and skill requirements
   (level 0–5, mandatory checkbox). **Edit** shows the saved requirements;
   **Delete** removes the project.
3. **Students:** add a student with skills and comma-separated preferences.
4. Invalid input (for example minimum capacity above maximum, or an unknown
   project in the preferences) is rejected with a message instead of an
   error page.
