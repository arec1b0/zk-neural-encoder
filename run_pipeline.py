import json
from zk_neural_encoder.pipeline.aot_orchestrator import AOTOrchestrator

def main():
    # Define a mock ABI for demonstration purposes
    mock_abi = [
        {"type": "uint256", "name": "totalSupply", "stateMutability": "view"},
        {"type": "uint256", "name": "balances", "stateMutability": "nonpayable"},
        {"type": "address", "name": "owner", "stateMutability": "view"},
        {"type": "function", "name": "transfer", "stateMutability": "nonpayable"}
    ]

    # Initialize orchestrator
    orchestrator = AOTOrchestrator()

    # Step 1: Train the agent (Optimization Phase)
    print("--- Starting Training Phase ---")
    orchestrator.run_training_pipeline(mock_abi, epochs=50)

    # Step 2: Generate the Manifest (Export Phase)
    print("\n--- Generating JSON Manifest ---")
    manifest = orchestrator.generate_manifest(mock_abi, output_path="encoding_manifest.json")

    # Output the resulting manifest
    print("\nFinal Encoding Manifest:")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()