# Third-party sources

Upstream benchmark repositories are intentionally not vendored.
`sources.lock.json` records the exact revisions used by this code release, and
`tools/bootstrap_third_party.sh` checks them out under the ignored
`third_party/src/` directory.

The SOTOPIA integration is reapplied after checkout:

- `overlays/sotopia/` adds the GRPO policy agent and mental/reward models;
- `patches/sotopia-agents-init.patch` exports the added SOTOPIA agent.

The upstream repositories retain their own licenses. Do not commit
`third_party/src/`; rerun the bootstrap script to reproduce it.
