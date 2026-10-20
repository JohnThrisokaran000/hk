@AGENTS.md

# Claude Code specifics

- Read `CONTEXT.md` and the top of `docs/plan/HANDOFF.md` at the start of a session; update HANDOFF before ending one.
- Plan mode is appropriate for anything touching the `.hk` format, the C ABI, or the backend interface.
- Prefer `git ls-files`/`find` over `ls` in this environment (the `ls` alias rejects directory arguments).
- Use the Agent tool only when the owner asks for it; the docs under `docs/plan/` and `docs/graph/` are designed so a
  fresh agent does not need a conversation to get oriented.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
