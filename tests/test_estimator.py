import pytest
from zk_neural_encoder.analyzer.static_analyzer import ContractFeature
from zk_neural_encoder.estimator.constraint_cost import ConstraintCostEstimator, EncodingType

def test_estimate_cost_bit_vector():
    feature = ContractFeature(name="test", data_type="uint256", size_bytes=32, is_mutable=True)
    cost = ConstraintCostEstimator.estimate_cost(feature, EncodingType.BIT_VECTOR)
    assert cost == 256  # 32 bytes * 8 bits

def test_estimate_cost_single_field():
    immutable_feature = ContractFeature(name="test_const", data_type="uint256", size_bytes=32, is_mutable=False)
    cost = ConstraintCostEstimator.estimate_cost(immutable_feature, EncodingType.SINGLE_FIELD)
    assert cost == 2
    
    mutable_feature = ContractFeature(name="test_var", data_type="uint256", size_bytes=32, is_mutable=True)
    cost = ConstraintCostEstimator.estimate_cost(mutable_feature, EncodingType.SINGLE_FIELD)
    assert cost == 15