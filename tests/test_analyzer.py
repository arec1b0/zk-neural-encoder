import pytest
from zk_neural_encoder.analyzer.static_analyzer import StaticAnalyzer

def test_parse_abi_extracts_correct_features():
    analyzer = StaticAnalyzer()
    mock_abi = [
        {"type": "function", "name": "transfer"},
        {"type": "uint256", "name": "balance", "stateMutability": "view"},
        {"type": "uint256", "name": "totalSupply", "stateMutability": "nonpayable"}
    ]
    
    features = analyzer.parse_abi(mock_abi)
    
    assert len(features) == 2
    
    balance_feature = next(f for f in features if f.name == "balance")
    assert balance_feature.size_bytes == 32
    assert balance_feature.is_mutable is False
    
    supply_feature = next(f for f in features if f.name == "totalSupply")
    assert supply_feature.size_bytes == 32
    assert supply_feature.is_mutable is True