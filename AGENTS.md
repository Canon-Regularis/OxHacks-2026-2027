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
| **Orchid** | Backend algorithm implementations, adapted behind GridPulse interfaces | Repository URL and local path pending; not available locally when this file was created |
| **WWW** | Main website template, page structure and layout reference | Team-provided reference: <https://kcpc.orthant.co.uk/>; source repository URL and local path pending |
| **Components** | Reusable styling and UI components for frontend polish | Repository URL and local path pending |
| **Existing exporter** | Fetch Elexon data and write CSVs for ingestion | Being implemented by a teammate; sample files, schema and code location pending |

The team plans to pull the reference repositories locally. Once available, record their exact paths, upstream URLs, revisions, licences and the files actually reused. Inspect compatibility and dependencies before choosing components. Preserve applicable attribution and licence notices.

## Orchid inventory supplied by the team

The shared integration-plan description names these five files:

- `spectral_template.py`
- `autoregressive_template.py`
- `forecasting_baseline_template.py`
- `smoothing_template.py`
- `state_space_template.py`

Referenced document: `Orchid_GridPulse_HRT_Algorithm_Architecture.md`. Only its title and summary were supplied in chat; the full document is not currently in this repository. The list above is not a verified inventory of Orchid's public symbols or dependencies.

**LightGBM status:** not verified. No Orchid implementation is available locally to inspect. The team's candidate list explicitly includes LightGBM classification and logistic regression, both of which require fitted models for predictive use. Library reuse alone does not eliminate fitting. If the MVP must avoid training, omit these methods unless suitable already-fitted artifacts exist and their applicability is established. No such artifacts have been confirmed.

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

## Integration agreements still needed

1. Obtain real CSV samples and agree identifiers, units, publication/delivery timestamps and revision handling.
2. Inspect Orchid and the full architecture document; select methods compatible with the no-training constraint and available data.
3. Agree algorithm adapters and frontend API response examples before parallel implementation.
4. Identify reusable WWW/Components files and preserve upstream attribution.
5. Keep credentials out of documentation, source control and frontend code. Use server-side configuration and placeholder examples only.

Preserve `(settlement_date, settlement_period)` identities, UTC instants and 46/48/50-period day support. Distinguish retrospective analysis from signals actually available at a historical cutoff. Describe scores according to the algorithm's supported meaning; do not label an arbitrary indicator as a probability.

## Repository layout

Follow [the repository structure and request flow](docs/gridpulse/REPOSITORY_STRUCTURE.md). Frontend code lives in `frontend/`; Python code in `backend/src/gridpulse/`; SQL migrations in `db/migrations/`; shared API examples in `contracts/`. Algorithms belong in `analysis/algorithms/` and are called through `analysis/service.py`. Keep HTTP, database access and pure calculations separate. Skeleton files are not implemented features.

## Related project documents

- [Overview](docs/gridpulse/README.md)
- [Data pipeline](docs/gridpulse/DATA_PIPELINE.md)
- [Algorithms and analysis contracts](docs/gridpulse/ALGORITHMS_AND_ANALYSIS.md)
- [Build and demo plan](docs/gridpulse/BUILD_AND_DEMO.md)
- [Extensions](docs/gridpulse/RESEARCH_AND_EXTENSIONS.md)
- [Sponsor reference](docs/sponsors.md)
