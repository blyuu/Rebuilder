"""Read RTAB-Map run log and database graph constraints for reproducibility."""

import argparse
import json
import re
import sqlite3
from pathlib import Path


def audit(database: Path, log: Path, output: Path, input_directory: Path) -> dict:
    with sqlite3.connect(database) as connection:
        nodes = connection.execute("SELECT id, stamp, hex(ground_truth_pose) FROM Node").fetchall()
        links = connection.execute("SELECT from_id, to_id, type FROM Link WHERE type IN (1,2,3)").fetchall()
    stamps = {node: stamp for node, stamp, _ in nodes}
    unique = {}
    for start, end, kind in links:
        key = (min(start, end), max(start, end), kind)
        unique[key] = {"from_node": key[0], "to_node": key[1], "type":
                       {1: "global", 2: "local_space", 3: "local_time"}[kind],
                       "from_timestamp_s": stamps.get(key[0]), "to_timestamp_s": stamps.get(key[1])}
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    rejected = []
    for line in lines:
        match = re.search(r"Loop closure (\d+)->(\d+) rejected!", line)
        if match:
            rejected.append({"from_node": int(match[1]), "to_node": int(match[2])})
    total = re.search(r"Total time=([\d.]+)s", "\n".join(lines))
    report = {"rtabmap_processing_time_s": float(total[1]) if total else None,
              "database_nodes": len(nodes),
              "ground_truth_input_file_present": (input_directory / "groundtruth.txt").exists(),
              "stored_ground_truth_pose_all_zero": all(set(bytes.fromhex(blob)) <= {0} for _, _, blob in nodes),
              "accepted_loop_constraints": list(unique.values()),
              "accepted_loop_constraint_count": len(unique),
              "rejected_loop_events": rejected, "rejected_loop_event_count": len(rejected)}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k not in ("accepted_loop_constraints", "rejected_loop_events")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("database", type=Path)
    p.add_argument("log", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("input_directory", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.database, a.log, a.output, a.input_directory), indent=2))
