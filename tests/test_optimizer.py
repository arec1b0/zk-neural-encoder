from zk_neural_encoder.optimizer.reinforcement_agent import ReinforcementOptimizer
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature
from zk_neural_encoder.estimator.constraint_cost import EncodingType

def test_optimizer_selects_encoding_and_calculates_cost():
    optimizer = ReinforcementOptimizer()
    feature = ContractFeature(name="test_balance", data_type="uint256", size_bytes=32, is_mutable=True)

    # Распаковываем 3 значения, игнорируя тензор log_prob
    encoding, cost, _ = optimizer.optimize_encoding(feature, deterministic=True)

    assert isinstance(encoding, EncodingType)
    assert isinstance(cost, int)
    assert cost > 0