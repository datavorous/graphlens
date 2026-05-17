from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph_parser import (
    LiteRTLMExtractor,
    TFLiteGraphParser,
    _find_dynamic_shape_ops,
    _report_dynamic_shape_ops,
)
from .partition_simulator import (
    OpSupport,
    PartitionResult,
    _load_op_support,
    _simulate_partition,
    _report_partitions,
    _report_seams,
)


class Inspector:
    """Class-based entry point for static partition diagnosis."""

    def __init__(self, model_path: str, op_support_path: str):
        self.model_path = model_path
        self._op_support_path = op_support_path

        self.graph: Optional[Dict[str, Any]] = None
        self.op_support: Optional[OpSupport] = None
        self.dynamic_shape_ops: Optional[List[Dict[str, Any]]] = None
        self.partitions: Optional[List[PartitionResult]] = None

    @classmethod
    def from_litertlm(
        cls,
        litertlm_path: str,
        op_support_path: str,
        dump_dir: str = "./dump",
    ) -> "Inspector":
        extracted = LiteRTLMExtractor.extract(litertlm_path, dump_dir)
        if not extracted:
            raise RuntimeError(f"No .tflite sections were extracted from: {litertlm_path}")
        return cls(model_path=extracted[0], op_support_path=op_support_path)

    @classmethod
    def from_tflite(cls, tflite_path: str, op_support_path: str) -> "Inspector":
        return cls(model_path=tflite_path, op_support_path=op_support_path)

    def analyze(self) -> None:
        parser = TFLiteGraphParser()
        self.graph = parser.parse(self.model_path)
        self.op_support = _load_op_support(self._op_support_path)
        self.dynamic_shape_ops = _find_dynamic_shape_ops(self.graph)
        self.partitions = _simulate_partition(self.graph, self.op_support)

    def _require_analysis(self) -> None:
        if self.graph is None or self.op_support is None or self.dynamic_shape_ops is None or self.partitions is None:
            raise RuntimeError("Inspector is not analyzed yet. Call analyze() first.")

    def report_dynamic_shapes(self, *args, **kwargs) -> None:
        self._require_analysis()
        _report_dynamic_shape_ops(self.graph, *args, **kwargs)

    def report_partitions(self, *args, **kwargs) -> None:
        self._require_analysis()
        _report_partitions(self.partitions, *args, **kwargs)

    def report_seams(self, *args, **kwargs) -> None:
        self._require_analysis()
        _report_seams(self.graph, self.partitions, *args, **kwargs)
