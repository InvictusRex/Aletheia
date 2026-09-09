# AGENTS.md — permanent working agreements (Aletheia repo)

Every agent and contributor working in this repository MUST follow these rules.
They outrank one-off prompt instructions when in conflict, except explicit
user overrides for the current task.

## 1. Code comments

- NEVER add `#`-style code comments to any file, in any language.
- NEVER add `"""` / `'''` docstrings or any other comment convention.
  No comments, period. Code must be self-explanatory through names
  and structure.
- When rewriting a region, remove any comments/docstrings inside it.
  Do not strip comments elsewhere as drive-by churn.

## 2. Git commits

- Format: `type(scope): short imperative subject` (lowercase, no period).
  Types: `feat`, `fix`, `refactor`, `chore`, `test`, `docs`.
- One focused commit per task. Stage only the task's files.
- NEVER commit: secrets, API keys, `.env` files, `veritas-main/`,
  model weights, logs, temp PDFs, or unrelated working-tree changes.
- NEVER push, amend history, or rewrite published commits unless the
  user explicitly asks. Backdate a commit only when explicitly asked.

## 3. Shell (Windows PowerShell 5.1)

- `grep`, `head`, `tail`, `rg`, `&&` do NOT exist here. Use the dedicated
  Grep/Read tools instead of shell text utilities.
- Chain dependent commands with `cmd1; if ($?) { cmd2 }`.
- Never `cd` inside commands; pass `workdir` to the shell tool.
- Background jobs do NOT survive across tool calls. For detached work
  use `Start-Process` with output redirection to a log file, then poll
  the log with `Get-Content -Tail`.

## 4. Verification

- Verify by running: compile checks, targeted tests, then the full
  suite (`py -3.12 -m pytest backend/tests -q --no-header -p no:cacheprovider`
  from the repo root). Report exact pass/fail counts and runtimes.
- Fix failures at their root cause. Never weaken tests to make them pass.
- Never claim a test passed unless it was actually executed.

## 5. Scope discipline

- No unrelated refactors. No architectural expansion. No new background
  infrastructure (queues, workers, brokers) unless explicitly requested.
- Respect frozen/in-progress areas named in the task prompt: do not
  modify, restructure, or commit them.
- Deterministic unit tests use fakes only at transport boundaries
  (HTTP/SDK). Never fake provider/model inference to claim validation.

## 6. Parallel work

- When working alongside other agents, write ONLY to the file set
  assigned to you. Never edit, move, or delete files outside it —
  report cross-cutting issues instead of fixing them yourself.

## 7. Database safety

- NEVER truncate, delete, drop, or reset an existing E2E/test database
  merely to start another run. Existing results are valuable data.
- Full-document E2E runs MUST NOT truncate the database before or after.
- Rerun via the resumable/idempotent pipeline instead.
- A clean database requires stated justification BEFORE any destructive
  operation, plus a verified full backup (record path and pre-operation
  document/fact/relationship/chunk counts) first.
- Never overwrite or delete the only known-good backup.
- Never auto-restore an older backup after a failure; first identify
  which backup matches the desired state.
- When in doubt whether data is valuable, STOP and ask instead of truncating.
- Truncation is NEVER an implicit part of an E2E workflow.
