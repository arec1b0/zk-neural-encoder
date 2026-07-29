import json
from pathlib import Path
from typing import List, Dict, Any
from zk_neural_encoder.analyzer.static_analyzer import StaticAnalyzer, ContractFeature
from zk_neural_encoder.optimizer.reinforcement_agent import ReinforcementOptimizer
from zk_neural_encoder.estimator.constraint_cost import EncodingType
from zk_neural_encoder.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class AOTOrchestrator:
    def __init__(self) -> None:
        self.analyzer = StaticAnalyzer()
        self.optimizer = ReinforcementOptimizer()
        logger.info("AOT Pipeline Orchestrator initialized.")

    def run_training_pipeline(self, abi_payload: List[Dict[str, Any]], epochs: int = 100) -> None:
        """
        Parses the ABI and executes the RL training loop.
        """
        logger.info("Starting AOT Training Phase.")
        features: List[ContractFeature] = self.analyzer.parse_abi(abi_payload)
        
        if not features:
            logger.warning("No mutable state variables found. Skipping training.")
            return
            
        self.optimizer.train_agent(features, epochs=epochs)

    def generate_manifest(self, abi_payload: List[Dict[str, Any]], output_path: str = "encoding_manifest.json") -> Dict[str, Any]:
        """
        Runs the trained agent deterministically and exports a JSON manifest for ZK compilers.
        """
        logger.info("Generating Encoding Manifest.")
        features: List[ContractFeature] = self.analyzer.parse_abi(abi_payload)
        
        manifest = {
            "version": "1.0",
            "generator": "Neural-Guided-AOT",
            "layouts": {}
        }
        
        total_constraints = 0
        
        for feature in features:
            # Deterministic inference for manifest generation
            encoding, cost, _ = self.optimizer.optimize_encoding(feature, deterministic=True)
            total_constraints += cost
            
            manifest["layouts"][feature.name] = {
                "type": feature.data_type,
                "encoding_strategy": encoding.value,
                "estimated_constraints": cost
            }
            
        manifest["summary"] = {
            "total_estimated_constraints": total_constraints,
            "variable_count": len(features)
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(manifest, f, indent=4)
            logger.info(f"Manifest successfully exported to {output_path}")
        except IOError as e:
            logger.error(f"Failed to write manifest: {e}")
            
        return manifest