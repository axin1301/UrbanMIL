import runpy
import sys
from pathlib import Path


def _normalize_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    drop_value_for = {
        "--use_sims",
        "--ablation_mode",
    }
    drop_flag_only = {
        "--run_all_ablations",
    }

    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg in drop_flag_only:
            continue

        if arg in drop_value_for:
            if idx + 1 < len(argv):
                skip_next = True
            continue

        if arg.startswith("--use_sims="):
            continue

        if arg.startswith("--ablation_mode="):
            continue

        cleaned.append(arg)

    cleaned.extend(["--use_sims", "sat,stv,inst"])
    return cleaned


def main() -> None:
    target = Path(__file__).with_name("train_ind_semretr_reasoning_mech.py")
    sys.argv = [str(target)] + _normalize_args(sys.argv[1:])
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
