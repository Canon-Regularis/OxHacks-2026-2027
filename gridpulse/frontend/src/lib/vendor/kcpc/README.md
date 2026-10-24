# KCPC source provenance

Selected source was adapted from `orthant-kcpc/components` revision `9626af5` (sibling checkout `components/`) using the team's supplied reference repositories. Homepage usage patterns were inspected in `orthant-kcpc/www` revision `b16a659`, `apps/kcpc/src/routes/+page.svelte`; card/section examples came from its About and Software pages.

Reused source paths (relative to Components `src/lib/`):

- `components/animations/rotating-earth/`: globe, world geometry and themed variant.
- `components/backgrounds/pixel-blast/`: homepage “work with us” effect, now on About.
- `components/text/text-shimmer/text-shimmer.svelte`: the accompanying animated About heading.
- `components/cards/evervault-card/`: committee mouse-following alphanumeric reveal used for decorative context.
- `components/primitives/card/card.svelte`: shared card primitive.
- `shadcn.css`: neutral theme tokens in `src/styles/theme.css`.

Imports of `#lib/utils.js` were rewritten to `$lib/vendor/kcpc/utils`, a small local `clsx`/`tailwind-merge` adapter. EvervaultCard has an optional `contentClass` extension so a wide context section can use the original reveal effect without its fixed circular badge layout. The application uses the original animation implementations, with a consumer-level reduced-motion switch and keyed globe resizing. PixelBlast is client-loaded only when WebGL2 is available; the main dashboard works without it. Heavy animations are separate lazy-loaded chunks.

These are vendor references, not original GridPulse algorithms. No explicit upstream licence file or package licence declaration was found in these checkouts; this record does not assert an open-source licence. Preserve attribution and confirm redistribution terms with the source owners before public distribution. Do not edit the sibling repos to run GridPulse.
