# ast-grep rules

These rules enforce repository-specific Rust conventions that Clippy cannot express precisely.

- Keep one test file per rule with the same filename and rule ID.
- Include both valid and invalid snippets, especially boundary and nesting cases.
- Prefer AST kinds, fields, and relationships; use regex only for leaf text or syntax without a
  useful node boundary.
- Add `files` or `ignores` when a rule only applies to part of the repository.
- Add automatic fixes only when the rewrite is unambiguous. Snapshot-test rules that have fixes;
  match-only rules use the lighter valid/invalid tests.

Run `ast-grep scan --filter RULE_ID` while developing an individual rule. Before committing, run
`make lint-ast test-ast`.
