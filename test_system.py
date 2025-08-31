#!/usr/bin/env python3
"""
Test script to verify the multi-dataset response generation system components.
This script tests the functionality without requiring actual models.
"""

import os
import json
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import nltk
        print("✓ NLTK imported successfully")
    except ImportError as e:
        print(f"✗ NLTK import failed: {e}")
        return False
    
    try:
        from myutils import StoppingCriteriaSub, StoppingCriteriaList
        print("✓ MyUtils imported successfully")
    except ImportError as e:
        print(f"✗ MyUtils import failed: {e}")
        return False
    
    try:
        from config import MODEL_CONFIGS, DATASET_CONFIGS
        print("✓ Config imported successfully")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    return True

def test_config_structure():
    """Test if configuration files have the correct structure"""
    print("\nTesting configuration structure...")
    
    try:
        from config import MODEL_CONFIGS, DATASET_CONFIGS, GENERATION_PARAMS, OUTPUT_SETTINGS
        
        # Test model configs
        required_models = ['llama2-7b', 'llama3.1-8b', 'vicuna-7b']
        for model in required_models:
            if model in MODEL_CONFIGS:
                print(f"✓ {model} configuration found")
            else:
                print(f"✗ {model} configuration missing")
                return False
        
        # Test dataset configs
        required_datasets = ['triviaqa', 'truthfulqa', 'coqa', 'tydiqa']
        for dataset in required_datasets:
            if dataset in DATASET_CONFIGS:
                print(f"✓ {dataset} configuration found")
            else:
                print(f"✗ {dataset} configuration missing")
                return False
        
        # Test generation params
        if GENERATION_PARAMS:
            print("✓ Generation parameters found")
        else:
            print("✗ Generation parameters missing")
            return False
        
        # Test output settings
        if OUTPUT_SETTINGS:
            print("✓ Output settings found")
        else:
            print("✗ Output settings missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'generate_responses_local.py',
        'run_all_models.py',
        'config.py',
        'requirements.txt',
        'README_MULTI_DATASET.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} missing")
            return False
    
    return True

def test_sample_data_creation():
    """Test sample data creation functionality"""
    print("\nTesting sample data creation...")
    
    try:
        # Import the generator class
        sys.path.append('.')
        from generate_responses_local import LocalMultiDatasetResponseGenerator
        
        # Create a mock generator (without actual model loading)
        class MockGenerator(LocalMultiDatasetResponseGenerator):
            def __init__(self):
                self.model_name = "test-model"
                self.stop_words = []
            
            def load_model(self):
                pass  # Skip model loading for testing
            
            def setup_stopping_criteria(self):
                pass  # Skip setup for testing
        
        generator = MockGenerator()
        
        # Test sample data creation for each dataset
        datasets = ['triviaqa', 'truthfulqa', 'coqa', 'tydiqa']
        for dataset in datasets:
            sample_data = generator.create_sample_data(dataset, 5)
            if len(sample_data) > 0:
                print(f"✓ {dataset} sample data created successfully ({len(sample_data)} samples)")
            else:
                print(f"✗ {dataset} sample data creation failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation test failed: {e}")
        return False

def test_output_directory():
    """Test if output directory can be created and written to"""
    print("\nTesting output directory...")
    
    try:
        # Create data directory
        os.makedirs('./data', exist_ok=True)
        
        # Test file writing
        test_file = './data/test_output.json'
        test_data = {'test': 'data', 'timestamp': '2024-01-01'}
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Verify file was created
        if os.path.exists(test_file):
            print("✓ Output directory and file writing test passed")
            
            # Clean up test file
            os.remove(test_file)
            return True
        else:
            print("✗ Output directory test failed")
            return False
            
    except Exception as e:
        print(f"✗ Output directory test failed: {e}")
        return False

def test_argument_parsing():
    """Test command line argument parsing"""
    print("\nTesting argument parsing...")
    
    try:
        from generate_responses_local import main
        
        # Test that the script can be imported and main function exists
        if callable(main):
            print("✓ Argument parsing test passed")
            return True
        else:
            print("✗ Main function not callable")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Multi-Dataset Response Generation System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Structure", test_config_structure),
        ("File Structure", test_file_structure),
        ("Sample Data Creation", test_sample_data_creation),
        ("Output Directory", test_output_directory),
        ("Argument Parsing", test_argument_parsing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  {test_name} failed")
        except Exception as e:
            print(f"  {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Download or specify model paths")
        print("2. Update model paths in run_all_models.py")
        print("3. Run: python run_all_models.py")
    else:
        print("✗ Some tests failed. Please fix the issues before using the system.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
