# GrowthBook — Open Issues to Hack On

## Python Stats Engine (gbstats package)

### [#401 — Bug: Error in stats engine when mean is negative](https://github.com/growthbook/growthbook/issues/401)
- **Labels:** roadmap
- The stats engine throws an error when a metric's mean is negative. Old issue, still open — likely subtle or hard to reproduce.
- **Stack:** Python (`packages/stats/gbstats`)
- **Why interesting:** Pure Python stats bug, testable locally with just `pytest`

### [#3902 — Feature: Add calculated SD to experiment page per metric](https://github.com/growthbook/growthbook/issues/3902)
- **Labels:** enhancement
- Show standard deviation alongside each metric in the experiment results page.
- **Stack:** Python stats engine + frontend display
- **Why interesting:** Stats + data science territory

### [#4505 — Feature: More stats engine settings in power calculator](https://github.com/growthbook/growthbook/issues/4505)
- **Labels:** power
- Expand the options available in the power calculator (e.g. more stats engine settings).
- **Stack:** Python (`packages/stats/gbstats/power/`)
- **Why interesting:** Power analysis is core data science / experimentation knowledge

---

## Good First Issues (labeled)

### [#3471 — Bug: Deleting SDK connections does not remove webhooks](https://github.com/growthbook/growthbook/issues/3471)
- **Labels:** bug, good first issue
- When you delete an SDK connection, the associated webhooks aren't cleaned up. Users hit their webhook limit with no way to recover.
- **Fix:** Find where SDK deletion happens in the backend and add webhook cleanup logic.
- **Stack:** TypeScript/Node backend

### [#2693 — Feature: Add project selector to create feature modal](https://github.com/growthbook/growthbook/issues/2693)
- **Labels:** enhancement, good first issue
- Right now you can only assign a feature to a project *after* creating it. This asks for adding a project dropdown to the creation modal.
- **Fix:** Add a project selector to the feature creation modal.
- **Stack:** React (frontend)

### [#2644 — Feature: Disable user registration when self-hosting](https://github.com/growthbook/growthbook/issues/2644)
- **Labels:** enhancement, good first issue
- Let self-hosted admins turn off open registration so only invited users can join.
- **Fix:** Add an env var / config flag on the backend and hide the register button on the frontend when set.
- **Stack:** Backend (Node) + Frontend (React)

---

## Slightly Harder but Interesting

### [#4799 — Bug: API returns deleted metrics in experiment results](https://github.com/growthbook/growthbook/issues/4799)
- **Labels:** bug
- Deleted metrics still show up in experiment results via the API.
- **Fix:** Likely a missing filter on deleted records in a SQL query or API response.
- **Why interesting:** Close to data/SQL territory — relevant to data engineering background.

### [#4950 — Bug: Bar chart y-axis starts at non-zero value](https://github.com/growthbook/growthbook/issues/4950)
- **Labels:** bug
- Classic misleading chart bug — y-axis doesn't start at zero, making small differences look huge.
- **Fix:** Frontend/visualization fix to force y-axis origin at zero.
- **Stack:** React / charting library

### [#4703 — Bug: Deleting experiment custom field also deletes feature custom field with same name](https://github.com/growthbook/growthbook/issues/4703)
- **Labels:** bug
- Two different entity types share a name and deletion logic accidentally deletes both.
- **Fix:** Backend scoping bug — scope the deletion query to the correct entity type.
- **Stack:** TypeScript/Node backend

---

## Recommended Issues (ranked by DS relevance)

### ~~1.~~ [#3202 — Feature: One-tail test in frequentist engine](https://github.com/growthbook/growthbook/issues/3202)
- **Stack:** Python (`packages/stats/gbstats`)
- **Why it's great:** Direct extension of the two-sided Welch's t-test we already built.
- **Status:** BLOCKED — maintainers want one-sided support in the sequential testing engine too, not just fixed-horizon. Scope is bigger than it looks.

### 1. [#3047 — Feature: Margin of error — show sample size needed](https://github.com/growthbook/growthbook/issues/3047)
- **Stack:** Python (`packages/stats/gbstats`)
- **Why it's great:** Classic DS math: solve `n = (z * sigma / E)²` for target widths.

### 2. [#4552 — Feature: Power calculator — exact sample sizes](https://github.com/growthbook/growthbook/issues/4552)
- **Stack:** Python (`packages/stats/gbstats/power/`)
- **Why it's great:** Computing exact N from power equations instead of rounding to weeks. Similar territory to #3047.

### 3. [#401 — Bug: Stats engine error on negative means](https://github.com/growthbook/growthbook/issues/401)
- **Stack:** Python (`packages/stats/gbstats`)
- **Why it's great:** Debugging the delta method when `mean_control < 0` — the `mean_control²` and `mean_control⁴` terms still work, but the lift formula `(t - c) / c` flips sign interpretation. Good bug hunt.

---

## Previous Recommendation
- **#4950** — already done! PR #5379 submitted
- **#3471** — well-defined cleanup bug, good TypeScript contribution
- **#4799** — deleted metrics in API, closest to data/SQL work
