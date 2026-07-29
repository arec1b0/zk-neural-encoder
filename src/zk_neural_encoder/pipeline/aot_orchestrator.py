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

    def run_pipeline(self, abi_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the full Ahead-Of-Time encoding optimization pipeline.
        """
        logger.info("Starting Neural-Guided Encoding optimization pass.")
        
        try:
            features: List[ContractFeature] = self.analyzer.parse_abi(abi_payload)
            if not features:
                logger.warning("No mutable state variables found in the ABI payload.")
                return {"status": "success", "results": []}

            results = []
            total_constraints = 0

            for feature in features:
                encoding, cost = self.optimizer.optimize_encoding(feature)
                total_constraints += cost
                
                results.append({
                    "feature_name": feature.name,
                    "data_type": feature.data_type,
                    "selected_encoding": encoding.value,
                    "constraint_cost": cost
                })

            logger.info(f"Pipeline completed. Total estimated constraints: {total_constraints}")
            
            return {
                "status": "success",
                "total_constraints": total_constraints,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}