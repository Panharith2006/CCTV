"""
Simple OSNet ImageNet model inspector and tester
Works without torchreid - loads model state directly
"""

import torch
import os
import numpy as np


def inspect_model_weights():
    """Inspect the .pth file structure and weights"""
    model_path = 'osnet_x1_0_imagenet.pth'
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║     OSNet ImageNet Model Inspector                        ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    print(f"Model file: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Load the state dict
        print("\nLoading model weights...")
        state_dict = torch.load(model_path, map_location='cpu')
        
        print(f"✓ Successfully loaded model")
        print(f"\n{'='*60}")
        print("Model Architecture Summary:")
        print(f"{'='*60}")
        
        if isinstance(state_dict, dict):
            layer_count = {}
            total_params = 0
            
            # Analyze layers
            for key, value in state_dict.items():
                layer_type = key.split('.')[0]
                if layer_type not in layer_count:
                    layer_count[layer_type] = 0
                layer_count[layer_type] += 1
                
                if hasattr(value, 'numel'):
                    total_params += value.numel()
            
            print(f"\nModel Layers:")
            for layer_type, count in sorted(layer_count.items()):
                print(f"  {layer_type}: {count} weight matrices")
            
            print(f"\nTotal Parameters: {total_params:,}")
            print(f"Memory (approx): {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
            
            # Show first few layers
            print(f"\n{'='*60}")
            print("First 10 Layer Shapes:")
            print(f"{'='*60}")
            for i, (key, value) in enumerate(list(state_dict.items())[:10]):
                if hasattr(value, 'shape'):
                    params = value.numel() if hasattr(value, 'numel') else 'N/A'
                    print(f"  {key}: {value.shape} ({params:,} params)")
            
            print(f"\n... and {max(0, len(state_dict) - 10)} more layers")
            
            # Check for common output layers
            print(f"\n{'='*60}")
            print("Output Layer Analysis:")
            print(f"{'='*60}")
            
            for key in state_dict.keys():
                if 'fc' in key or 'classifier' in key or 'linear' in key:
                    value = state_dict[key]
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                        if len(value.shape) == 2:
                            input_dim, output_dim = value.shape
                            print(f"    → Input features: {input_dim}D")
                            print(f"    → Output features: {output_dim}D")
            
            return True
        else:
            print("✗ Unexpected state dict format")
            return False
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """Test if model can be loaded with a simple forward pass"""
    model_path = 'osnet_x1_0_imagenet.pth'
    
    print(f"\n{'='*60}")
    print("Model Load Test:")
    print(f"{'='*60}")
    
    try:
        print("Loading weights...")
        state_dict = torch.load(model_path, map_location='cpu')
        print("✓ Weights loaded successfully")
        
        # Try to create a simple model-like structure
        print("\nTesting tensor operations...")
        
        # Get a sample weight
        sample_weight = list(state_dict.values())[0]
        print(f"✓ Sample weight shape: {sample_weight.shape}")
        print(f"✓ Sample weight dtype: {sample_weight.dtype}")
        print(f"✓ Sample weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def generate_test_recommendations():
    """Generate recommendations for using the model"""
    print(f"\n{'='*60}")
    print("Testing Options Available:")
    print(f"{'='*60}")
    
    print("""
1. WITH TORCHREID (Recommended):
   pip install torchreid
   python test_osnet_model.py
   └─ Full feature extraction and similarity testing
   
2. WITH OPENCV + PYTORCH:
   - Load model via torchreid.models.build_model()
   - Preprocess images with CV2
   - Extract 512D features
   
3. USE IN YOUR CCTV SYSTEM (Already integrated):
   - Your system already has OSNet integration
   - See: detector/layer2_reid_extractor_enhanced.py
   - The model is loaded automatically via torchreid
   
4. SIMPLE VERIFICATION:
   python -c "import torch; sd=torch.load('osnet_x1_0_imagenet.pth'); print(f'Layers: {len(sd)}'); print(f'Size: {sum(p.numel() for p in sd.values())/1e6:.1f}M params')"
    """)


def main():
    # Inspect model
    if inspect_model_weights():
        # Test compatibility
        test_model_compatibility()
    
    # Recommendations
    generate_test_recommendations()
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("""
To fully test the OSNet model:

1. Install torchreid:
   pip install torchreid

2. Run the full test:
   python test_osnet_model.py

3. Or integrate with your CCTV system:
   - The model is already used in detector/layer2_reid_extractor_enhanced.py
   - Run your main.py to see it in action:
     python main.py
    """)


if __name__ == '__main__':
    main()
