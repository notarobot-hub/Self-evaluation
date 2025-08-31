#!/usr/bin/env python3
"""
Batch script to run response generation for all models on multiple datasets.
This script will process Llama 2 7B, Llama 3.1 8B, and Vicuna 7B models.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_model_generation(model_name, model_path, samples_per_dataset=50, device="cuda"):
    """Run response generation for a specific model"""
    print(f"\n{'='*60}")
    print(f"Starting generation for {model_name}")
    print(f"Model path: {model_path}")
    print(f"Samples per dataset: {samples_per_dataset}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist.")
            print("Please ensure the model is downloaded before running.")
            return False
        
        # Run the generation script
        cmd = [
            sys.executable, "generate_responses_local.py",
            "--model_name", model_name,
            "--model_path", model_path,
            "--samples_per_dataset", str(samples_per_dataset),
            "--device", device
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully completed generation for {model_name}")
            print("Output:", result.stdout)
            return True
        else:
            print(f"Error running generation for {model_name}")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Exception occurred while running {model_name}: {e}")
        return False

def main():
    """Main function to run all models"""
    print("Multi-Model Response Generation Script")
    print("This script will generate responses for all models on multiple datasets")
    
    # Configuration
    models = {
        "llama2-7b": {
            "path": "./models/llama2-7b-chat-hf",  # Update this path
            "samples": 50
        },
        "llama3.1-8b": {
            "path": "./models/llama3.1-8b-instruct",  # Update this path
            "samples": 50
        },
        "vicuna-7b": {
            "path": "./models/vicuna-7b-v1.5",  # Update this path
            "samples": 50
        }
    }
    
    # Check if CUDA is available
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    except ImportError:
        device = "cpu"
        print("PyTorch not available, using CPU")
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Run generation for each model
    successful_models = []
    failed_models = []
    
    for model_name, config in models.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        success = run_model_generation(
            model_name=model_name,
            model_path=config["path"],
            samples_per_dataset=config["samples"],
            device=device
        )
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Wait a bit between models to free up memory
        if model_name != list(models.keys())[-1]:  # Not the last model
            print("Waiting 30 seconds before next model...")
            time.sleep(30)
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful models: {len(successful_models)}")
    for model in successful_models:
        print(f"  ✓ {model}")
    
    print(f"Failed models: {len(failed_models)}")
    for model in failed_models:
        print(f"  ✗ {model}")
    
    if successful_models:
        print(f"\nResults saved in ./data/ directory")
        print("Files generated:")
        for model in successful_models:
            print(f"  - {model}_all_datasets_responses.json")
            print(f"  - {model}_triviaqa_responses.json")
            print(f"  - {model}_truthfulqa_responses.json")
            print(f"  - {model}_coqa_responses.json")
            print(f"  - {model}_tydiqa_responses.json")
    
    print(f"\n{'='*60}")
    if failed_models:
        print("Some models failed. Please check the error messages above.")
        print("Make sure the model paths are correct and models are downloaded.")
    else:
        print("All models completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
