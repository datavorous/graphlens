# Run from the project root: uv run python examples/main.py
# Expects gemma.litertlm two levels up (edge/gemma.litertlm).
from graphlens import Inspector


def section(title: str) -> None:
    print(f"\n## {title}\n")


inspector = Inspector.from_litertlm(
    "../../gemma.litertlm",
    op_support_path="../analysis/opSupportMap.csv",
    dump_dir="../../litertlm_dump",
)
inspector.analyze()

section("Dynamic shape / index inputs (fragmentation candidates)")
inspector.report_dynamic_shapes()

section("Partition simulation")
inspector.report_partitions()

section("CPU seams (+/-2 ops around each CPU partition)")
inspector.report_seams(context=2, kind="CPU")
