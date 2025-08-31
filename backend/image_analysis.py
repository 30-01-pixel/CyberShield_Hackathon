import json
import logging
import os
import tempfile
import requests
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from model_manager import get_model_manager

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def prepare_image_input(img_path: str) -> str:
    """
    Takes either a local image path or a URL and returns local file path.
    Downloads remote images to temporary files.
    """
    if os.path.exists(img_path):  # Local file
        return img_path
    else:  # Assume it's a URL, download it
        try:
            response = requests.get(img_path, timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(response.content)
            temp_file.close()
            
            logger.info(f"Downloaded image from {img_path} to {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading image from {img_path}: {e}")
            raise

def validate_image(img_path: str) -> bool:
    """
    Validate that the image file exists and is readable.
    """
    try:
        if not os.path.exists(img_path):
            return False
            
        # Try to open with PIL to validate it's a valid image
        with Image.open(img_path) as img:
            img.verify()  # Verify it's a valid image
        
        return True
    except Exception as e:
        logger.error(f"Image validation failed for {img_path}: {e}")
        return False

def image_analysis_llm_block(img_path: str) -> Dict[str, Any]:
    """
    Analyze image using Sparrow VLM for anti-India content detection.
    Currently returns placeholder results - full Sparrow VLM integration pending.
    
    Args:
        img_path: Path to image file or URL
        
    Returns:
        Dict containing is_anti_india (bool), confidence (float), reasoning (str)
    """
    try:
        logger.info(f"Starting image analysis for: {img_path}")
        
        # Prepare image (download if URL)
        local_image_path = prepare_image_input(img_path)
        
        # Validate image
        if not validate_image(local_image_path):
            return {
                "is_anti_india": False,
                "confidence": 0.0,
                "reasoning": "Invalid or corrupted image file"
            }
        
        # Get image info for logging
        with Image.open(local_image_path) as img:
            width, height = img.size
            format_type = img.format
            logger.info(f"Image info: {width}x{height}, format: {format_type}")
        
        # TODO: Integrate with Sparrow VLM
        # For now, return placeholder result
        logger.info("Using placeholder image analysis - Sparrow VLM integration pending")
        
        # Get model manager for future integration
        manager = get_model_manager()
        result = manager.analyze_image_for_anti_india(local_image_path)
        
        # Clean up temporary file if it was downloaded
        if not os.path.exists(img_path) and local_image_path != img_path:
            try:
                os.unlink(local_image_path)
                logger.info(f"Cleaned up temporary file: {local_image_path}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary file {local_image_path}: {e}")
        
        logger.info(f"Image analysis complete. Result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        return {
            "is_anti_india": False,
            "confidence": 0.0,
            "reasoning": f"Image analysis failed: {str(e)}"
        }

if __name__ == "__main__":
    # Test the image analysis
    test_image = "anti-India.jpg"  # Local test image
    
    if os.path.exists(test_image):
        print(f"Testing image analysis with: {test_image}")
        result = image_analysis_llm_block(test_image)
        print(f"Analysis result: {json.dumps(result, indent=2)}")
    else:
        print(f"Test image {test_image} not found. Skipping test.")
        
        # Test with a placeholder
        print("Testing with placeholder analysis...")
        result = image_analysis_llm_block("test_image.jpg")  # Non-existent image
        print(f"Analysis result: {json.dumps(result, indent=2)}")
