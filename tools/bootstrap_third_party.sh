#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${ROOT}/third_party/src"

declare -A URLS=(
  [sotopia]="https://github.com/sotopia-lab/sotopia.git"
  [bigtom]="https://github.com/cicl-stanford/procedural-evals-tom.git"
  [fantom]="https://github.com/skywalker023/fantom.git"
  [hitom]="https://github.com/ying-hui-he/Hi-ToM_dataset.git"
  [opentom]="https://github.com/seacowx/OpenToM.git"
  [tomi]="https://github.com/facebookresearch/ToMi.git"
  [aida]="https://github.com/The-Inscrutable-X/AIDoubleAgentDefenders.git"
)

declare -A REVISIONS=(
  [sotopia]="80aeaaa3af6ba8e9dec506672fc673043a4fec37"
  [bigtom]="fe647d680bddb69f738519313bed625f9e93b549"
  [fantom]="1cae6fa30f5ba04ca0fff5f5716b5ba7055e2e85"
  [hitom]="4279d3f783ff4f3b9fcced2a2fec9f6328683f82"
  [opentom]="3f22b66276b2d7ca5fe573c28c79cc0d077aafc5"
  [tomi]="dea2bca9b366c41cdab0bd717353cc02453e193e"
  [aida]="0f1ad2d336d9986eed736a7aa458275beb1535fc"
)

ALL_SOURCES=(sotopia bigtom fantom hitom opentom tomi aida)

usage() {
  cat <<'EOF'
Usage: tools/bootstrap_third_party.sh [all|SOURCE ...]

Sources: sotopia bigtom fantom hitom opentom tomi aida
With no arguments, all pinned sources are installed.
EOF
}

apply_patch_once() {
  local checkout="$1"
  local patch="$2"
  if git -C "${checkout}" apply --check "${patch}" 2>/dev/null; then
    git -C "${checkout}" apply "${patch}"
  elif git -C "${checkout}" apply --reverse --check "${patch}" 2>/dev/null; then
    echo "  patch already applied: $(basename "${patch}")"
  else
    echo "Cannot apply or verify patch ${patch} in ${checkout}." >&2
    exit 1
  fi
}

checkout_source() {
  local name="$1"
  local destination="${SOURCE_ROOT}/${name}"
  local revision="${REVISIONS[${name}]}"

  if [[ ! -d "${destination}/.git" ]]; then
    echo "Cloning ${name}..."
    mkdir -p "${destination}"
    git -C "${destination}" init -q
    git -C "${destination}" remote add origin "${URLS[${name}]}"
    if ! git -C "${destination}" fetch --depth 1 origin "${revision}"; then
      git -C "${destination}" fetch origin
    fi
    git -C "${destination}" checkout --detach -q "${revision}"
  else
    local current
    current="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${current}" != "${revision}" ]]; then
      if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
        echo "Refusing to change dirty checkout: ${destination}" >&2
        exit 1
      fi
      git -C "${destination}" fetch origin "${revision}"
      git -C "${destination}" checkout --detach -q "${revision}"
    fi
  fi

  if [[ "$(git -C "${destination}" rev-parse HEAD)" != "${revision}" ]]; then
    echo "Revision verification failed for ${name}." >&2
    exit 1
  fi

  case "${name}" in
    sotopia)
      apply_patch_once "${destination}" "${ROOT}/third_party/patches/sotopia-agents-init.patch"
      cp -R "${ROOT}/third_party/overlays/sotopia/." "${destination}/"
      ;;
    aida)
      apply_patch_once "${destination}" "${ROOT}/third_party/patches/aida-local.patch"
      ;;
  esac
  echo "Ready: ${name} @ ${revision}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
  REQUESTED=("${ALL_SOURCES[@]}")
else
  REQUESTED=("$@")
fi

mkdir -p "${SOURCE_ROOT}"
for name in "${REQUESTED[@]}"; do
  if [[ -z "${URLS[${name}]+x}" ]]; then
    echo "Unknown source: ${name}" >&2
    usage >&2
    exit 2
  fi
  checkout_source "${name}"
done
