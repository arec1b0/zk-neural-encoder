import torch
import torch.nn as nn
from typing import Tuple
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature
from zk_neural_encoder.estimator.constraint_cost import EncodingType, ConstraintCostEstimator
from zk_neural_encoder.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class EncodingPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReinforcementOptimizer:
    def __init__(self) -> None:
        self.encodings = list(EncodingType)
        # Input features: size_bytes, is_mutable (mapped to 2 dimensions)
        self.policy = EncodingPolicyNetwork(input_dim=2, output_dim=len(self.encodings))
        logger.info("Initialized ReinforcementOptimizer (Policy Network)")

    def _feature_to_tensor(self, feature: ContractFeature) -> torch.Tensor:
        return torch.tensor([float(feature.size_bytes), float(feature.is_mutable)], dtype=torch.float32)

    def optimize_encoding(self, feature: ContractFeature) -> Tuple[EncodingType, int]:
        """
        Uses the policy network to select an encoding and calculates the cost.
        In a full AOT compilation loop, this acts as the environment step.
        """
        state = self._feature_to_tensor(feature)
        
        with torch.no_grad():
            probabilities = self.policy(state)
            
        # Select action with highest probability (greedy selection for AOT inference)
        action_idx = torch.argmax(probabilities).item()
        selected_encoding = self.encodings[action_idx]
        
        cost = ConstraintCostEstimator.estimate_cost(feature, selected_encoding)
        logger.info(f"Selected encoding for '{feature.name}': {selected_encoding.value} | Constraints: {cost}")
        
        return selected_encoding, cost