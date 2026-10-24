# GridPulse agent instructions and project context

Use this file as repository-wide guidance when working on GridPulse. It records team decisions, intended reuse and information still to confirm. Referenced repositories and algorithms are not integrated merely because they are listed here.

## Project and scope

- **Event:** OxHack 2026; 25-hour build window; three developers.
- **Project:** GridPulse, a GB electricity-market analysis dashboard covering wind forecasts, demand forecasts and imbalance settlement prices.
- **Target track:** HRT's Best Use of Data for Predictions & Decision Making, as supplied by the team; confirm submission requirements against event materials.
- **MVP:** existing API-to-CSV exporter → backend ingestion → Tiger Data → backend algorithms → backend API → polished frontend dashboard.
- **Current constraint:** no ML model training in the MVP. Reuse existing algorithm implementations where suitable, with their actual inputs, outputs and limitations documented.
- **Hosting:** Vultr for the application.
- **Main extension:** a backend Gemini API agent that uses data and analysis tools to investigate results and answer higher-level questions.
- **Optional sponsor exploration:** Snowflake API, with its useful role still to be decided. It is not a core dependency.

## Intended code and design sources

| Source | Intended use | Location / status |
|---|---|---|
| **Orchid** | Backend algorithm implementations, adapted behind GridPulse interfaces | [Orchid](https://github.com/Orthant-Research-Group/orchid), local `../orchid/`, cloned revision `ca8fc92` |
| **WWW** | Main website template, page structure and layout reference | [WWW](https://github.com/orthant-kcpc/www), local `../www/`, cloned revision `b16a659`; KCPC app: `apps/kcpc/` |
| **Components** | Reusable styling and UI components for frontend polish | [Components](https://github.com/orthant-kcpc/components), local `../components/`, cloned revision `9626af5` |
| **Orchid** | Backend algorithm implementations, adapted behind GridPulse interfaces | <https://github.com/Orthant-Research-Group/orchid> (private), commit `ca8fc9290722fb45ab27e022b76365beba1306c8`; P0 modules vendored in `gridpulse/backend/src/gridpulse/analysis/algorithms/orchid/`. No licence file upstream: confirm terms before this repository is made public |
| **WWW** | Main website template, page structure and layout reference | Team-provided reference: <https://kcpc.orthant.co.uk/>; source repository URL and local path pending |
| **Components** | Reusable styling and UI components for frontend polish | Repository URL and local path pending |
| **Existing exporter** | Fetch Elexon data and write CSVs for ingestion | Being implemented by a teammate; sample files, schema and code location pending |

The team plans to pull the reference repositories locally. Once available, record their exact paths, upstream URLs, revisions, licences and the files actually reused. Inspect compatibility and dependencies before choosing components. Preserve applicable attribution and licence notices.

## Orchid modules in use

The Orchid → GridPulse catalogue and architecture are in issue #3. Its P0 methods (issue #6) come from four Orchid modules, vendored verbatim. Their SHA-256 digests are recorded in `orchid/__init__.py`:

| Vendored file | Upstream path (`algorithms/core/…`) |
|---|---|
| `autoregressive.py` | `forecasting/autoregressive/autoregressive_template.py` |
| `forecasting_baseline.py` | `forecasting/baseline/forecasting_baseline_template.py` |
| `smoothing.py` | `forecasting/smoothing/smoothing_template.py` |
| `spectral.py` | `spectral/spectral_template.py` |

Each module's upstream `_stress_test.py` is ported to `gridpulse/backend/tests/analysis/orchid/`, with only the import lines changed. GridPulse code calls Orchid only through the adapters in `analysis/algorithms/`, which use one canonical implementation per method; see [algorithms and analysis contracts](docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md). `state_space_template.py` is left for the P1 work (#7). Do not edit vendored files: change Orchid upstream and re-vendor.

**LightGBM status:** the vendored modules contain no LightGBM or other classifier, and the P0 port adds none. The team's candidate list explicitly includes LightGBM classification and logistic regression, both of which require fitted models for predictive use. Library reuse alone does not eliminate fitting. If the MVP must avoid training, omit these methods unless suitable already-fitted artifacts exist and their applicability is established. No such artifacts have been confirmed.
**LightGBM status:** not verified. Orchid is now cloned locally, but its relevant implementations and dependencies have not yet been audited for GridPulse. The team's candidate list explicitly includes LightGBM classification and logistic regression, both of which require fitted models for predictive use. Library reuse alone does not eliminate fitting. If the MVP must avoid training, omit these methods unless suitable already-fitted artifacts exist and their applicability is established. No such artifacts have been confirmed.

## Candidate algorithms and scope checks

These are proposals from the team, not a list of implemented features.

| Candidate | Scope note |
|---|---|
| Point-in-time ASOF joins | Data preparation; select records available at a declared historical cutoff |
| Sliding-window feature engineering | Deterministic signals from available wind/demand history |
| Logistic regression | Requires fitted coefficients; outside the no-training MVP unless an appropriate fitted model is supplied |
| LightGBM classification | Requires a trained model; same scope restriction |
| Walk-forward cross-validation | Evaluation procedure; normally involves repeated fitting when evaluating trainable models |
| Probability calibration | Usually fits a mapping on held-out predictions; defer under the current constraint |
| Precision–recall and threshold optimisation | Metrics require meaningful scores and labels; threshold tuning uses historical outcomes and needs separate evaluation |
| Cost-sensitive decision optimisation | Requires defined costs and meaningful scores; validate any policy tuned on historical data separately |
| Feature ablation | Tests contributions under a defined analysis protocol; model-based ablation commonly requires refitting and does not prove causality |
| KNN historical similarity and cohort analysis | Can retrieve comparable historical periods without fitting a predictive classifier; define scaling, distance, eligible history and minimum cohort size |

Other proposed extensions include a probabilistic scenario engine, an autonomous research agent and decision stress testing. These are ideas for after the core flow works. State-space, autoregressive and related statistical methods may also estimate parameters; inspect actual implementations before assuming they require no fitting.

## Confirmed frontend and map direction

**Confirmed frontend scope:** Overview, Explore and Methods are MVP pages. The immediate implementation focus is Overview only. Investigate/Gemini and the synoptic weather map remain extensions.


Follow [FRONTEND_DESIGN.md](docs/gridpulse/FRONTEND_DESIGN.md). Preserve KCPC responsiveness/theme and requested globe/fade effects; use committee-style motion where appropriate and Software-style cards for sectioned content. Reference repos are selective reuse sources, not a requirement to copy the whole site.

The Gemini extension should support animated view changes during investigations. The synoptic-map extension should represent wind, pressure, fronts and isobars correctly and allow a separate UK electricity-use/forecast overlay. Exact tools, weather providers and regional-data sources are not selected. Do not promote proposed layouts, API recommendations or scenario-engine ideas into confirmed scope.

## Integration agreements still needed

1. Obtain real CSV samples and agree identifiers, units, publication/delivery timestamps and revision handling.
2. Orchid P0 methods are integrated (#6). Select P1–P3 methods (#7, #8, #10) that are compatible with the no-training constraint and the available data.
3. Agree algorithm adapters and frontend API response examples before parallel implementation.
4. Identify reusable WWW/Components files and preserve upstream attribution.
5. Keep credentials out of documentation, source control and frontend code. Use server-side configuration and placeholder examples only.

Preserve `(settlement_date, settlement_period)` identities, UTC instants and 46/48/50-period day support. Distinguish retrospective analysis from signals actually available at a historical cutoff. Describe scores according to the algorithm's supported meaning; do not label an arbitrary indicator as a probability.

## Repository layout

Follow [the repository structure and request flow](docs/gridpulse/REPOSITORY_STRUCTURE.md). Frontend code lives in `gridpulse/frontend/`; Python code in `gridpulse/backend/src/gridpulse/`; SQL migrations in `gridpulse/db/migrations/`; shared API examples in `gridpulse/contracts/`. Algorithms belong in `analysis/algorithms/` and are called through `analysis/service.py`. Keep HTTP, database access and pure calculations separate. Skeleton files are not implemented features.

## Related project documents

- [Frontend design brief](docs/gridpulse/FRONTEND_DESIGN.md)
- [Overview](docs/gridpulse/README.md)
- [Data pipeline](docs/gridpulse/DATA_PIPELINE.md)
- [Algorithms and analysis contracts](docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md)
- [Build and demo plan](docs/gridpulse/BUILD_AND_DEMO.md)
- [Extensions](docs/gridpulse/RESEARCH_AND_EXTENSIONS.md)
- [Sponsor reference](docs/sponsors.md)
