from enum import Enum
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature

class EncodingType(Enum):
    SINGLE_FIELD = "SINGLE_FIELD"
    BIT_VECTOR = "BIT_VECTOR"
    LIMB_DECOMPOSITION = "LIMB_DECOMPOSITION"

class ConstraintCostEstimator:
    @staticmethod
    def estimate_cost(feature: ContractFeature, encoding: EncodingType) -> int:
        """
        Calculates a deterministic constraint cost for a specific encoding.
        These represent ZK gate counts in an arithmetized circuit (e.g., Halo2).
        """
        if encoding == EncodingType.SINGLE_FIELD:
            # Cheap to store, expensive to range-check if mutable
            return 2 if not feature.is_mutable else 15
        elif encoding == EncodingType.BIT_VECTOR:
            # 1 constraint per bit for boolean logic
            return feature.size_bytes * 8
        elif encoding == EncodingType.LIMB_DECOMPOSITION:
            # e.g., 4 limbs for a 256-bit (32 byte) integer
            return (feature.size_bytes // 8) * 3
        else:
            raise ValueError(f"Unknown encoding type: {encoding}")