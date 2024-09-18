# Convert PyTorch model to ONNX format

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch

PATH = "Thalirajesh/Aerial-Drone-Image-Segmentation"
SHAPE = (704, 1056)

def main():
    
    # Load model directly
    model = SegformerForSemanticSegmentation.from_pretrained(PATH)
    
    dummy_input = torch.zeros(1, 3, SHAPE[0], SHAPE[1])
    torch.onnx.export(model, dummy_input, "ariel-segmentation-unet.onnx")

if __name__ == "__main__":
    main()