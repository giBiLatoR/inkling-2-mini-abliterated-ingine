"""Compare exact output IDs on a controlled short prompt with natural and deliberately wrong drafts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROMPT = "List the counting numbers from one to fifty separated by commas. Begin with 1."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--wrong-drafts-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.wrong_drafts_child:
        import dspark
        import run

        dspark.Drafter.propose = lambda self, anchor, count: [77777] * count
        sys.argv = ["run.py", str(args.model_dir), PROMPT, "--tokens", "32", "--dspark", "4"]
        run.main()
        return

    runner = Path(__file__).with_name("run.py")
    cmd = [sys.executable, str(runner), str(args.model_dir), PROMPT, "--tokens", "32", "--dspark"]

    def receipt(argv):
        result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=240)
        return json.loads(result.stderr.splitlines()[-1])

    serial = receipt(cmd + ["0"])
    natural = receipt(cmd + ["4"])
    wrong = receipt([sys.executable, __file__, str(args.model_dir), "--wrong-drafts-child"])
    assert natural["accepted"] > 0
    assert wrong["accepted"] == 0 and wrong["proposed"] > 0
    assert serial["output_id_sha256"] == natural["output_id_sha256"] == wrong["output_id_sha256"]
    print("Exact 32-token match with accepted proposals and forced rollback.")


if __name__ == "__main__":
    main()
