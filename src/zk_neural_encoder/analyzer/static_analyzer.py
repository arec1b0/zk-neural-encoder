from pydantic import BaseModel, Field
from typing import List, Dict, Any
from zk_neural_encoder.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class ContractFeature(BaseModel):
    name: str
    data_type: str
    size_bytes: int
    is_mutable: bool = Field(default=True)

class StaticAnalyzer:
    def __init__(self) -> None:
        logger.info("Initializing StaticAnalyzer")

    def parse_abi(self, abi_data: List[Dict[str, Any]]) -> List[ContractFeature]:
        """
        Parses smart contract ABI to extract deterministic structure.
        """
        features = []
        for item in abi_data:
            if item.get("type") in ("function", "constructor", "event", "error", "fallback", "receive"):
                continue
            
            name = item.get("name", "unknown")
            data_type = item.get("type", "unknown")
            
            # Simple heuristic for byte size
            size_bytes = 32 if "256" in data_type else 8
            
            # State mutability heuristic for public storage variables
            is_mutable = item.get("stateMutability", "nonpayable") != "view"
            
            feature = ContractFeature(
                name=name,
                data_type=data_type,
                size_bytes=size_bytes,
                is_mutable=is_mutable
            )
            features.append(feature)
            
        logger.info(f"Extracted {len(features)} deterministic features from ABI.")
        return features