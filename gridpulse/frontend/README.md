# Frontend work area

MVP pages: **Overview, Explore, Methods**. Build **Overview first**; current work should not expand into other pages. Investigate/Gemini and weather mapping remain extensions.

Follow the [confirmed frontend design brief](../../docs/gridpulse/FRONTEND_DESIGN.md) for reuse patterns and extension boundaries. Reference checkouts now exist at workspace level in `../www/` and `../components/` relative to the repository root.

Frontend/product owner works here. Framework setup is pending inspection of WWW and Components; no runnable frontend or dependency manifest exists yet. Preserve this responsibility split when adding framework entry points and build configuration.

- `src/pages/`: dashboard screens and loading/error/result state.
- `src/components/ui/`: reusable primitives adapted from Components.
- `src/components/charts/`: chart presentation.
- `src/features/`: cohesive feature code for overview, analysis and later agent UI.
- `src/lib/api.ts`: shared HTTP boundary; add request functions here when endpoints are agreed.
- `src/styles/`: shared styles and design tokens.
- `public/`: static assets with appropriate attribution.

Use [contract examples](../contracts/examples/) during parallel development, visibly labelled as fixtures. Read the [request flow](../../docs/gridpulse/REPOSITORY_STRUCTURE.md#request-flow) before connecting the API. Database credentials and Gemini keys belong only on the backend.

Record exact install/development/build commands here after choosing the framework. Configure a same-origin `/api` proxy or an explicit public backend URL; public frontend configuration must contain no secrets.
