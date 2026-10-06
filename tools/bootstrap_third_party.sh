#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${ROOT}/third_party/src"

declare -A URLS=(
  [sotopia]="https://github.com/sotopia-lab/sotopia.git"
  [bigtom]="https://github.com/cicl-stanford/procedural-evals-tom.git"
  [fantom]="https://github.com/skywalker023/fantom.git"
  [tomi]="https://github.com/facebookresearch/ToMi.git"
)

declare -A REVISIONS=(
  [sotopia]="80aeaaa3af6ba8e9dec506672fc673043a4fec37"
  [bigtom]="fe647d680bddb69f738519313bed625f9e93b549"
  [fantom]="1cae6fa30f5ba04ca0fff5f5716b5ba7055e2e85"
  [tomi]="dea2bca9b366c41cdab0bd717353cc02453e193e"
)

ALL_SOURCES=(sotopia bigtom fantom tomi)

# FANToM's released data is fetched by its own loader (task/dataset_loader.py);
# the same archive and SHA-256 are used here so no Python environment is needed.
FANTOM_DATA_URL="https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"
FANTOM_DATA_SHA256="1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4"

usage() {
  cat <<'EOF'
Usage: tools/bootstrap_third_party.sh [all|SOURCE ...]

Sources: sotopia bigtom fantom tomi
With no arguments, all pinned sources are installed. The ToMi test split is
extracted from its pinned archive and the FANToM data is downloaded and
checksum-verified, so both transfer evaluations are ready to run.
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

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

extract_tomi_data() {
  local checkout="$1"
  local archive="${checkout}/tomi_balanced_story_types.zip"
  if [[ -f "${checkout}/tomi_balanced_story_types/fb_all_test.txt" ]]; then
    echo "  data already extracted: tomi_balanced_story_types/"
    return
  fi
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "${archive}" -d "${checkout}"
  else
    python3 -m zipfile -e "${archive}" "${checkout}"
  fi
  echo "  extracted tomi_balanced_story_types/"
}

fetch_fantom_data() {
  local checkout="$1"
  local data_dir="${checkout}/data/fantom"
  local archive="${checkout}/data/fantom.tar.gz"
  if [[ -f "${data_dir}/fantom_v1.json" ]]; then
    echo "  data already present: data/fantom/fantom_v1.json"
    return
  fi
  mkdir -p "${data_dir}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${FANTOM_DATA_URL}" -o "${archive}"
  else
    wget -q "${FANTOM_DATA_URL}" -O "${archive}"
  fi
  if [[ "$(sha256_of "${archive}")" != "${FANTOM_DATA_SHA256}" ]]; then
    rm -f "${archive}"
    echo "FANToM data checksum mismatch for ${FANTOM_DATA_URL}." >&2
    exit 1
  fi
  tar -xzf "${archive}" -C "${data_dir}"
  rm -f "${archive}"
  # Same marker FANToM's loader writes, so it does not download again.
  printf '%s\n%s' "$(date '+%Y-%m-%d %H:%M:%S')" "1.0" > "${data_dir}/.built"
  echo "  fetched data/fantom/fantom_v1.json"
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
    tomi)
      extract_tomi_data "${destination}"
      ;;
    fantom)
      fetch_fantom_data "${destination}"
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
