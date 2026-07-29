import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import mlflow
from mlflow.models import infer_signature
import time
from typing import Tuple, List, Dict, Any
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
    def __init__(self, learning_rate: float = 1e-3) -> None:
        self.encodings = list(EncodingType)
        self.policy = EncodingPolicyNetwork(input_dim=2, output_dim=len(self.encodings))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        logger.info("Initialized ReinforcementOptimizer (REINFORCE Agent)")

    def _feature_to_tensor(self, feature: ContractFeature) -> torch.Tensor:
        return torch.tensor([float(feature.size_bytes), float(feature.is_mutable)], dtype=torch.float32)

    def optimize_encoding(self, feature: ContractFeature, deterministic: bool = True) -> Tuple[EncodingType, int, torch.Tensor]:
        """
        Selects an encoding. If deterministic is False, it samples from the distribution (for training).
        Returns the encoding, cost, and the log probability of the selected action.
        """
        state = self._feature_to_tensor(feature)
        probabilities = self.policy(state)
        
        m = Categorical(probabilities)
        if deterministic:
            action_idx = torch.argmax(probabilities).item()
        else:
            action_idx = m.sample().item()
            
        selected_encoding = self.encodings[action_idx]
        cost = ConstraintCostEstimator.estimate_cost(feature, selected_encoding)
        log_prob = m.log_prob(torch.tensor(action_idx))
        
        return selected_encoding, cost, log_prob

    def train_agent(self, features: List[ContractFeature], epochs: int = 100) -> None:
        """
        Executes the REINFORCE training loop and tracks metrics via MLflow.
        """
        mlflow.set_experiment("Neural_Encoding_Optimization")
        
        with mlflow.start_run():
            mlflow.log_param("learning_rate", self.optimizer.param_groups[0]['lr'])
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("num_features", len(features))
            
            logger.info(f"Starting training loop for {epochs} epochs.")
            
            for epoch in range(epochs):
                log_probs = []
                rewards = []
                epoch_constraints = 0
                baseline_constraints = 0
                
                for feature in features:
                    # Baseline: Static compiler typically chooses Limb Decomposition for everything
                    baseline_cost = ConstraintCostEstimator.estimate_cost(feature, EncodingType.LIMB_DECOMPOSITION)
                    baseline_constraints += baseline_cost
                    
                    encoding, cost, log_prob = self.optimize_encoding(feature, deterministic=False)
                    epoch_constraints += cost
                    
                    # Reward is the reduction in constraints (positive is good)
                    reward = baseline_cost - cost
                    
                    log_probs.append(log_prob)
                    rewards.append(reward)

                # Calculate constraint reduction percentage
                reduction_pct = 0.0
                if baseline_constraints > 0:
                    reduction_pct = ((baseline_constraints - epoch_constraints) / baseline_constraints) * 100.0

                # REINFORCE update step
                policy_loss = []
                for log_prob, reward in zip(log_probs, rewards):
                    policy_loss.append(-log_prob * reward)
                
                self.optimizer.zero_grad()
                loss = torch.stack(policy_loss).sum()
                loss.backward()
                self.optimizer.step()
                
                # Logging metrics to MLflow
                mlflow.log_metric("loss", loss.item(), step=epoch)
                mlflow.log_metric("total_constraints", epoch_constraints, step=epoch)
                mlflow.log_metric("constraint_reduction_pct", reduction_pct, step=epoch)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Reduction: {reduction_pct:.2f}%")
            
            # Практичный обход сломанного PT2-экспортера в MLflow
            try:
                import os
                model_path = "policy_network.pth"
                torch.save(self.policy.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="model_weights")
                os.remove(model_path) # Очищаем локальный мусор
                logger.info("Training complete. Model weights logged to MLflow as artifact.")
            except Exception as e:
                logger.error(f"Model logging to MLflow failed: {e}", exc_info=True)
                raise