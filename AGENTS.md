# AGENTS.md — Awesome ML in Plasma Physics

## Repository Overview

A curated awesome-list of machine learning resources for plasma physics, tokamaks, and stellarators. The list is maintained in `README.md` with paper entries auto-generated from a BibTeX file.

## Repository Structure

- `README.md` — The awesome list. **Do not hand-edit the Research Papers section.** Use the generator script instead.
- `plasma_physics_ml_bibliography.bib` — BibTeX bibliography. Add new paper entries here.
- `add_papers_to_readme.py` — Regenerates the Research Papers section of README from the bib file.
- `logo.png` — List logo.

## README Section Order

The README has the following structure (preserve this order):

1. Logo + title + Awesome badge
2. Contents — table of contents with section descriptions
3. Tools (manually maintained)
   - Simulation and Modeling Frameworks
   - Machine Learning Frameworks for Plasma Physics
   - Data Platforms, Datasets & Benchmarks
   - GitHub Discovery and Organizations
4. Implementation Papers (manually maintained, papers with public code)
5. Research Papers (auto-generated from bib — **do not edit by hand**)
6. Contributing

## How to Add a New Paper

1. Add the entry to `plasma_physics_ml_bibliography.bib` with `keywords = {plasma-physics-ml}`.
2. Run `uv run python add_papers_to_readme.py` to regenerate the Research Papers section.
3. Review the diff, commit, and open a PR.

## How to Add a New Tool

Tools are maintained manually in the README. Add the entry to the appropriate subsection under `## Tools` following the existing format:

```
- [Name](URL) - Short description
```

## How to Add an Implementation Paper

Add manually under `## Implementation Papers` following the format:

```
- **Title** - *Authors (Year)* - [Paper](URL) | [Code](URL) - Short description
```

## Branch Protection

The `main` branch requires pull requests. All changes go through PR (squash merge preferred).

## Conventions

- Paper entries in Research Papers are marked with `<!-- imported-from-bib -->` — the script uses this to track which entries it manages.
- Manually added entries (without the marker) are preserved across script runs.
- Abstracts are truncated to ~500 characters with `...` appended.
- gh CLI: `gh pr view/edit/list` may fail with GraphQL errors. Use `gh api repos/OWNER/REPO/pulls/N` instead.
