2.2.1 (2026-10-31)
==================

New Features
------------

- Wheels are now published for Linux arm64 and for free-threaded CPython 3.14 and 3.15, on which gaussfit-rs runs without the GIL. One wheel per platform now serves CPython 3.12 and every later version. (`#38 <https://github.com/LM-SAL/gaussfit-rs/pull/38>`__)


Bug Fixes
---------

- Building gaussfit-rs from source now needs Rust 1.99 or later; macOS 27 can refuse to load an extension stripped by Rust 1.97 or older ("mis-aligned LINKEDIT string pool"). (`#38 <https://github.com/LM-SAL/gaussfit-rs/pull/38>`__)


Documentation
-------------

- The user guide's examples now run as written and are tested with the docs. (`#37 <https://github.com/LM-SAL/gaussfit-rs/pull/37>`__)


Release notes for 2.2.0 and earlier are on the `GitHub releases page <https://github.com/LM-SAL/gaussfit-rs/releases>`__.
