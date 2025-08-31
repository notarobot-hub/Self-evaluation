#!/usr/bin/env python3
"""
Example usage script for the multi-dataset response generation system.
This script demonstrates the workflow with sample data.
"""

import os
import json
from pathlib import Path

def create_sample_concepts():
    """Create sample concepts for demonstration"""
    sample_concepts = {
        "triviaqa": [
            {"question": "What is the capital of France?", "concept": "France"},
            {"question": "Who wrote Romeo and Juliet?", "concept": "Shakespeare"},
            {"question": "What is the chemical symbol for gold?", "concept": "Gold"},
            {"question": "In which year did World War II end?", "concept": "1945"},
            {"question": "What is the largest planet in our solar system?", "concept": "Jupiter"}
        ],
        "truthfulqa": [
            {"question": "What causes rain?", "concept": "Rain"},
            {"question": "How do plants grow?", "concept": "Plants"},
            {"question": "Why is the sky blue?", "concept": "Sky"},
            {"question": "What makes a rainbow?", "concept": "Rainbow"},
            {"question": "How do birds fly?", "concept": "Birds"}
        ],
        "coqa": [
            {"question": "What is the main character's name?", "concept": "Character"},
            {"question": "Where does the story take place?", "concept": "Story"},
            {"question": "What happens at the end?", "concept": "End"},
            {"question": "Who is the villain?", "concept": "Villain"},
            {"question": "What is the conflict?", "concept": "Conflict"}
        ],
        "tydiqa": [
            {"question": "What is the weather like?", "concept": "Weather"},
            {"question": "Where is the nearest hospital?", "concept": "Hospital"},
            {"question": "How do I get there?", "concept": "Directions"},
            {"question": "What time does it open?", "concept": "Time"},
            {"question": "How much does it cost?", "concept": "Cost"}
        ]
    }
    return sample_concepts

def simulate_forward_generation(concept):
    """Simulate forward response generation"""
    # This is a mock function - in reality, this would use the actual model
    sample_responses = {
        "France": "France is a country located in Western Europe, known for its rich history, culture, and cuisine. It is the largest country in the European Union by land area and has been a major power in Europe for centuries.",
        "Shakespeare": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language. He wrote approximately 39 plays and 154 sonnets.",
        "Gold": "Gold is a chemical element with the symbol Au and atomic number 79. It is a bright, slightly reddish yellow, dense, soft, malleable, and ductile metal.",
        "1945": "1945 was the final year of World War II, marked by the defeat of Nazi Germany in May and the surrender of Japan in September after the atomic bombings of Hiroshima and Nagasaki.",
        "Jupiter": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined."
    }
    
    return sample_responses.get(concept, f"{concept} is a concept that can be explained in detail with various aspects and characteristics.")

def simulate_backward_generation(forward_text, concept):
    """Simulate backward response generation"""
    # This is a mock function - in reality, this would use the actual model
    # The score represents how well the model can identify the concept from the explanation
    import random
    
    # Simulate varying familiarity scores
    base_score = 0.8  # Base familiarity
    variation = random.uniform(-0.2, 0.2)  # Add some randomness
    score = max(0.0, min(1.0, base_score + variation))
    
    return concept, score

def demonstrate_workflow():
    """Demonstrate the complete workflow"""
    print("Multi-Dataset Response Generation - Example Workflow")
    print("=" * 60)
    
    # Create sample concepts
    sample_concepts = create_sample_concepts()
    
    # Create output directory
    os.makedirs('./data', exist_ok=True)
    
    # Process each dataset
    all_results = []
    
    for dataset_name, concepts in sample_concepts.items():
        print(f"\nProcessing {dataset_name.upper()} dataset...")
        print("-" * 40)
        
        dataset_results = []
        
        for i, item in enumerate(concepts):
            question = item['question']
            concept = item['concept']
            
            print(f"Sample {i+1}: {question}")
            print(f"Concept: {concept}")
            
            # Step 1: Forward Generation
            print("  → Generating forward response...")
            forward_response = simulate_forward_generation(concept)
            print(f"  → Forward response: {forward_response[:100]}...")
            
            # Step 2: Backward Generation
            print("  → Generating backward response...")
            backward_response, backward_score = simulate_backward_generation(forward_response, concept)
            print(f"  → Backward response: {backward_response}")
            print(f"  → Familiarity score: {backward_score:.3f}")
            
            # Store results
            result = {
                'dataset': dataset_name,
                'question': question,
                'concept': concept,
                'forward_response': forward_response,
                'backward_response': backward_response,
                'backward_score': backward_score
            }
            
            dataset_results.append(result)
            all_results.append(result)
            
            print()
        
        # Save dataset-specific results
        dataset_file = f"./data/example_{dataset_name}_responses.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        print(f"✓ {dataset_name} results saved to {dataset_file}")
    
    # Save all results
    all_results_file = "./data/example_all_datasets_responses.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ All results saved to {all_results_file}")
    
    # Display summary
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(all_results)}")
    print(f"Datasets processed: {len(sample_concepts)}")
    print(f"Average familiarity score: {sum(r['backward_score'] for r in all_results) / len(all_results):.3f}")
    
    # Show sample output
    print(f"\nSample output structure:")
    if all_results:
        sample = all_results[0]
        print(json.dumps(sample, indent=2))
    
    print(f"\n{'='*60}")
    print("Example workflow completed successfully!")
    print("This demonstrates how the system would work with actual models.")
    print("To use with real models:")
    print("1. Download the required models")
    print("2. Update model paths in run_all_models.py")
    print("3. Run: python run_all_models.py")
    print(f"{'='*60}")

def show_file_structure():
    """Show the expected file structure"""
    print("\nExpected File Structure:")
    print("=" * 40)
    
    expected_files = [
        "generate_responses_local.py - Main generation script",
        "run_all_models.py - Batch processing script",
        "config.py - Configuration file",
        "requirements.txt - Dependencies",
        "README_MULTI_DATASET.md - Documentation",
        "test_system.py - System testing script",
        "example_usage.py - This example script",
        "myutils.py - Utility functions (from original project)",
        "data/ - Output directory (created automatically)"
    ]
    
    for file_info in expected_files:
        print(f"  {file_info}")

def main():
    """Main function"""
    print("Multi-Dataset Response Generation System")
    print("Example Usage and Demonstration")
    print("=" * 60)
    
    # Show file structure
    show_file_structure()
    
    # Demonstrate workflow
    demonstrate_workflow()
    
    return 0

if __name__ == "__main__":
    exit(main())
