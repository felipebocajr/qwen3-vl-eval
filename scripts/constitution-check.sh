#!/usr/bin/env bash
# Constitution Check Gate
# Scans specs, plans, and source code for constitutional violations.
# Exit 0 = pass, Exit 1 = fail
#
# Usage: ./scripts/constitution-check.sh [--specs-only] [--code-only] [--verbose]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECS_DIR="$REPO_ROOT/specs"
SRC_DIR="$REPO_ROOT/src"
CONSTITUTION="$REPO_ROOT/.specify/memory/constitution.md"

VERBOSE=false
CHECK_SPECS=true
CHECK_CODE=true
FOUND_VIOLATION=false

for arg in "$@"; do
    case "$arg" in
        --verbose) VERBOSE=true ;;
        --specs-only) CHECK_CODE=false ;;
        --code-only) CHECK_SPECS=false ;;
    esac
done

log() {
    if [[ "$VERBOSE" == true ]]; then
        echo "$1"
    fi
}

fail() {
    echo "CONSTITUTION VIOLATION: $1"
    echo "  Detected: $2"
    echo "  All model generation MUST use structured outputs (Pydantic/Outlines)."
    echo "  Regex-based parsing and unconstrained HF generation are strictly forbidden."
    FOUND_VIOLATION=true
}

# ---------------------------------------------------------------------------
# Gate 1: Structured Output Compliance — Plan/Tasks Scan
# ---------------------------------------------------------------------------
if [[ "$CHECK_SPECS" == true ]] && [[ -d "$SPECS_DIR" ]]; then
    log "[Gate 1] Scanning implementation plans (plan.md, tasks.md) for forbidden keywords..."

    # Forbidden keywords in implementation plans (case-insensitive).
    # We scan plan.md and tasks.md only — these are implementation artifacts,
    # not the original user requirements in spec.md.
    SPEC_FORBIDDEN=(
        'regex.*pars'
        're\.search'
        're\.match'
        're\.findall'
        'batch_decode'
        'remove.*outlines'
        'drop.*outlines'
        'remove.*pydantic'
        'drop.*pydantic'
        'unstructured.*output'
        'unconstrained.*generation'
        'string generation.*answer'
    )

    find "$SPECS_DIR" -type f \( -name 'plan.md' -o -name 'tasks.md' \) -print0 2>/dev/null \
        | while IFS= read -r -d '' file; do
        for pattern in "${SPEC_FORBIDDEN[@]}"; do
            if grep -inE "$pattern" "$file" >/dev/null 2>&1; then
                match=$(grep -inE "$pattern" "$file" | head -n 1)
                fail "Structured Output Rule (Section 6)" "$file: $match"
            fi
        done
    done
fi

# ---------------------------------------------------------------------------
# Gate 1: Structured Output Compliance — Code Scan
# ---------------------------------------------------------------------------
if [[ "$CHECK_CODE" == true ]] && [[ -d "$SRC_DIR" ]]; then
    log "[Gate 1] Scanning src/ for forbidden patterns (existing code warnings)..."

    CODE_TARGET_FILES=(
        "$SRC_DIR/parser.py"
        "$SRC_DIR/model.py"
        "$SRC_DIR/pipeline.py"
    )

    for file in "${CODE_TARGET_FILES[@]}"; do
        [[ -f "$file" ]] || continue

        # Check for regex-based extraction in parser/model/pipeline
        if grep -nE 're\.(search|match|findall|compile)' "$file" >/dev/null 2>&1; then
            match=$(grep -nE 're\.(search|match|findall|compile)' "$file" | head -n 1)
            echo "WARNING: Existing code violates Structured Output Rule (Section 6)"
            echo "  Detected: $file: $match"
            echo "  This is pre-amendment technical debt. Remediate when refactoring."
            echo ""
        fi

        # Check for unconstrained generate() without structured constraints
        if grep -nE '\.generate\(' "$file" >/dev/null 2>&1; then
            if ! grep -iE '(outlines|pydantic|guided_decoding|constraints)' "$file" >/dev/null 2>&1; then
                match=$(grep -nE '\.generate\(' "$file" | head -n 1)
                echo "WARNING: Existing code uses unconstrained generation (Section 6)"
                echo "  Detected: $file: $match"
                echo "  This is pre-amendment technical debt. Remediate when refactoring."
                echo ""
            fi
        fi
    done
fi

# ---------------------------------------------------------------------------
# Gate 2: Resumability Preservation (basic check)
# ---------------------------------------------------------------------------
if [[ "$CHECK_CODE" == true ]] && [[ -f "$SRC_DIR/pipeline.py" ]]; then
    log "[Gate 2] Checking resumability boundaries..."
    # Ensure trajectories.jsonl is still referenced for resume logic
    if ! grep -q 'trajectories.jsonl' "$SRC_DIR/pipeline.py"; then
        echo "WARNING: Gate 2 — trajectories.jsonl not referenced in pipeline.py; resumability may be broken."
    fi
fi

# ---------------------------------------------------------------------------
# Gate 3: Zero-Crash Preservation (basic check)
# ---------------------------------------------------------------------------
if [[ "$CHECK_CODE" == true ]] && [[ -f "$SRC_DIR/pipeline.py" ]]; then
    log "[Gate 3] Checking zero-crash boundaries..."
    # Ensure try/except exists in pipeline
    if ! grep -q 'try:' "$SRC_DIR/pipeline.py"; then
        echo "WARNING: Gate 3 — no try/except found in pipeline.py; zero-crash policy may be violated."
    fi
fi

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
if [[ "$FOUND_VIOLATION" == true ]]; then
    echo ""
    echo "FAIL: Constitution Check Gate detected one or more violations."
    echo "Refer to .specify/memory/constitution.md Section 6 and 7 for remediation guidance."
    exit 1
else
    log "PASS: Constitution Check Gate — all checks passed."
    exit 0
fi
