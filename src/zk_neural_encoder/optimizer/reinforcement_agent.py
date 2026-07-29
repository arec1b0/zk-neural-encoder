import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import mlflow
import logging
import os
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature
from zk_neural_encoder.estimator.constraint_cost import ConstraintCostEstimator, EncodingType

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, num_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

class ReinforcementOptimizer:
    def __init__(self, input_dim: int = 3, num_actions: int = 2):
        self.model = ActorCritic(input_dim, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # PPO Hyperparameters
        self.clip_eps = 0.2
        self.c_value = 0.5
        self.c_entropy = 0.01
        self.ppo_epochs = 4

        logger.info("Initialized ReinforcementOptimizer (PPO Agent)")

    def _feature_to_tensor(self, feature: ContractFeature) -> torch.Tensor:
        type_val = 1.0 if "uint" in feature.data_type else 0.0
        return torch.tensor([float(feature.size_bytes), type_val, float(feature.is_mutable)])

    def optimize_encoding(self, feature: ContractFeature, deterministic: bool = False):
        state = self._feature_to_tensor(feature).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.model(state)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = dist.sample().item()
                
            log_prob = dist.log_prob(torch.tensor(action))
            
        strategy = EncodingType.SINGLE_FIELD if action == 0 else EncodingType.LIMB_DECOMPOSITION
        cost = ConstraintCostEstimator.estimate_cost(feature, strategy)
        return strategy, cost, log_prob

    def train_agent(self, features: list[ContractFeature], epochs: int = 50):
        mlflow.set_experiment("Neural_Encoding_Optimization")
        
        with mlflow.start_run():
            mlflow.log_params({
                "algorithm": "PPO_SingleStep",
                "learning_rate": 1e-3,
                "clip_eps": self.clip_eps,
                "entropy_coef": self.c_entropy
            })

            for epoch in range(epochs):
                total_loss = 0.0
                
                try:
                    for feature in features:
                        state = self._feature_to_tensor(feature).unsqueeze(0)
                        
                        # Rollout
                        with torch.no_grad():
                            logits, old_value = self.model(state)
                            dist = Categorical(logits=logits)
                            action = dist.sample()
                            old_log_prob = dist.log_prob(action)
                        
                        strategy = EncodingType.SINGLE_FIELD if action.item() == 0 else EncodingType.LIMB_DECOMPOSITION
                        cost = ConstraintCostEstimator.estimate_cost(feature, strategy)
                        reward = torch.tensor([-float(cost)])
                        advantage = reward - old_value.detach()

                        # PPO Update
                        for _ in range(self.ppo_epochs):
                            logits, value = self.model(state)
                            dist = Categorical(logits=logits)
                            log_prob = dist.log_prob(action)
                            entropy = dist.entropy()

                            ratio = torch.exp(log_prob - old_log_prob)
                            surr1 = ratio * advantage
                            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
                            
                            actor_loss = -torch.min(surr1, surr2).mean()
                            critic_loss = nn.functional.mse_loss(value, reward.unsqueeze(0))
                            
                            loss = actor_loss + self.c_value * critic_loss - self.c_entropy * entropy.mean()

                            self.optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.optimizer.step()
                            
                            total_loss += loss.item()

                    mlflow.log_metric("loss", total_loss / len(features), step=epoch)
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss:.4f}")

                except Exception as e:
                    logger.error(f"Training failed during epoch {epoch}: {e}", exc_info=True)
                    raise

            try:
                model_path = "policy_network.pth"
                torch.save(self.model.state_dict(), model_path)
                mlflow.log_artifact(model_path, artifact_path="model_weights")
                if os.path.exists(model_path):
                    os.remove(model_path)
                logger.info("Training complete. Model weights logged to MLflow as artifact.")
            except Exception as e:
                logger.error(f"Model logging to MLflow failed: {e}", exc_info=True)
                raise