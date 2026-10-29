# Agent notes

Conventions you cannot read off the code. Install, test and docs commands live in `README.rst`, `tox.ini` and `pytest.ini`; the last section holds the rules for this repository.

## Lean changes

- Make the smallest change that fully solves the task: every caller, test and doc it touches, and nothing nobody asked for (options, abstractions, code "for later").
- Reuse before writing: search this package first, then its installed dependencies. A fix that belongs in a library upstream is proposed there, and the pull request says so.
- Deleting code beats adding it; code you move keeps its validation and error handling.
- Comments give the why in one line.

## Before a commit

- `prek run --files <changed files>` (or `pre-commit` with the same arguments) and the tests of every subpackage you touched pass.
- A bug fix comes with a test that fails before the fix and passes after it.
- A new public function or option comes with at least one test.

## Tests

- Tests live beside their code in `<subpackage>/tests/test_*.py` and test behavior users see.
- Every warning is an error. Fix the cause of a new warning; when the warning is expected, filter it at the narrowest scope, with a comment saying why: on the test (`@pytest.mark.filterwarnings("ignore:<message>:<Category>")`), on its file (`pytestmark = pytest.mark.filterwarnings(...)`), or in `pytest.ini` when many files hit it.
- Examples in docstrings and `.rst` files run as doctests.

## Docs

- Docstrings and docs are as short as the reader needs: a summary line, the parameters, and the behavior the code does not show. They say what the code does and what the user needs; how it works something out, where a rule came from, why a dependency exists and what is planned go in code comments or the pull request.
- The docs build is nitpicky and fails on warnings. Single backticks make a cross-reference that must resolve (`int`, `astropy.units.Quantity`, a public object of this package); parameter names, literals and file names take double backticks.

## Commits and pull requests

- Branch from `main`, and keep the branch current by rebasing on `main`, then `git push --force-with-lease`.
- Each commit is one focused piece of work.
- Commit and pull request titles are one sentence in the imperative that says what changed and, when it is not obvious, why: `Pass only the fitted samples to np.spacing in the float32 rounding test, as it warns about NaN on Windows`.
- A change users can see gets a towncrier fragment, with a type from `changelog/README.rst`: one or two full sentences for users. Its number is the pull request's, so it is added once the pull request is open.

## Before a pull request

- Review your own diff: every caller of a changed function still works, the tests asked for above exist, and the diff holds no debug output, personal paths or unrelated changes.
- Open the pull request when the branch is ready to merge: the checks before a commit and the whole test run pass locally.
- The description has four headed parts:
  - **What changed**
  - **Why**, linking the issue or pull request when one exists
  - **How it was verified**: the commands run and their results
  - **Needs a person**: each item from the list below that the change touches, or "none"
- Then watch CI, and reproduce a failed job locally with the same `tox -e <job>`.

## A person decides

An agent proposes these and lists them under **Needs a person**; a person makes the call:

- default values, scientific constants, thresholds and line lists, each with a citation or a data file
- physics or algorithm choices
- public API: names, signatures and return values of public functions
- names and formats of files written to disk, which people's analysis code reads
- dependency versions in `pyproject.toml`
- anything that deletes or overwrites data
- releases: tags, publishing and release metadata

## gaussfit-rs

- The remote is GitHub (`gh`). Pull requests are merged with a merge commit and keep every commit, so a later fix, style or lint change to work already committed goes into that commit: `git commit --fixup <commit>`, then `git rebase --autosquash main`.
- Fragments are `changelog/<PR>.<type>.rst`, with a counter for several of one type (`123.feature.1.rst`).
- Development commands, including the C reference and the figure tests, are in "Development" in `docs/installation.rst`.
- The Rust core is in `src/`, with its tests in `src/tests/` (`cargo test`). `pip install -e ".[dev]"` builds it through maturin; run that again after a Rust change, before testing from Python.
- `docs/design-notes.rst` holds one decision per section and names the code and tests that pin it; a change to those updates the section.
- `test_c_parity.py` compares fits with MUSE's C extension: the recorded fixtures always, and the extension live once `pip install ./third_party/c_reference` (C compiler needed) is in place. CI runs the live tests on Linux (`tox -e py313-cparity`); on macOS a few seeds fail against the Linux-recorded results, which is not a bug.
- Visual output is tested with the `figure_test` helper in `python/gaussfit_rs/tests/helpers.py`. The figure hash library is made on Linux by CircleCI, so figure tests fail on macOS; that is not a bug.
- `third_party/rmpfit/` is a patched copy of the `rmpfit` crate; a change to it follows "How to update" in `third_party/rmpfit/VENDORED.md`.
