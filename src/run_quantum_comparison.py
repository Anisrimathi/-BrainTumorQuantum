
from pathlib import Path
import json
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_hybrid.py"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

STRATEGIES = ["linear", "ring", "fully_connected"]


def main():
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Training script not found: {TRAIN_SCRIPT}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "purpose": "Quantum entanglement strategy pilot comparison",
        "warning": (
            "Pilot results only; not final clinical or "
            "full-volume segmentation results."
        ),
        "experiments": [],
    }

    for strategy in STRATEGIES:
        print("\n" + "=" * 65)
        print(f"Starting strategy: {strategy}")
        print("=" * 65, flush=True)

        # Each experiment gets its own log file.
        log_path = RESULTS_DIR / f"training_{strategy}_pilot.log"
        start = time.time()

        # The training script reads the strategy from an environment
        # variable only if configured to do so. To avoid silently
        # changing model code, create a temporary strategy-specific
        # copy of the script with its ENTANGLEMENT setting replaced.
        source = TRAIN_SCRIPT.read_text(encoding="utf-8")
        old_setting = 'ENTANGLEMENT = "ring"'
        new_setting = f'ENTANGLEMENT = "{strategy}"'

        if old_setting not in source:
            raise RuntimeError(
                "Could not find the expected ENTANGLEMENT setting "
                "in train_hybrid.py. No experiment was started."
            )

        modified_source = source.replace(old_setting, new_setting, 1)
        temporary_script = RESULTS_DIR / f"_train_{strategy}_pilot.py"
        temporary_script.write_text(
            modified_source,
            encoding="utf-8",
        )

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    [sys.executable, str(temporary_script)],
                    cwd=str(PROJECT_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        finally:
            temporary_script.unlink(missing_ok=True)

        elapsed = time.time() - start

        experiment = {
            "strategy": strategy,
            "exit_code": result.returncode,
            "elapsed_seconds": round(elapsed, 2),
            "log": str(log_path),
            "status": (
                "completed"
                if result.returncode == 0
                else "failed"
            ),
        }
        summary["experiments"].append(experiment)

        print(json.dumps(experiment, indent=2), flush=True)

        if result.returncode != 0:
            print(
                f"Experiment {strategy} failed. "
                f"Inspect {log_path}. Stopping to avoid "
                "presenting an incomplete comparison as complete."
            )
            break

    summary_path = RESULTS_DIR / "quantum_strategy_pilot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nComparison run summary:")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()
