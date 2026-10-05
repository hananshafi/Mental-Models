# Third-party sources

Upstream benchmark and simulator repositories are intentionally not vendored.
`sources.lock.json` records the exact revisions used by this code release, and
`tools/bootstrap_third_party.sh` checks them out under the ignored
`third_party/src/` directory.

Two local integrations are reapplied after checkout:

- `overlays/sotopia/` adds the GRPO policy agent and mental/reward models.
- `patches/sotopia-agents-init.patch` exports the added SOTOPIA agent.
- `patches/aida-local.patch` makes the local AIDA evaluation path usable without
  mandatory Weights & Biases or Azure settings.

The upstream repositories retain their own licenses. Do not commit
`third_party/src/`; rerun the bootstrap script to reproduce it.
