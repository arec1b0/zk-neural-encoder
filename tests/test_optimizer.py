import pytest
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature
from zk_neural_encoder.optimizer.reinforcement_agent import ReinforcementOptimizer

def test_optimizer_selects_encoding_and_calculates_cost():
    optimizer = ReinforcementOptimizer()
    feature = ContractFeature(name="test_balance", data_type="uint256", size_bytes=32, is_mutable=True)
    
    encoding, cost = optimizer.optimize_encoding(feature)
    
    assert encoding in optimizer.encodings
    assert isinstance(cost, int)
    assert cost > 0