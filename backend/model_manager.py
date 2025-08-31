import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import requests
import pandas as pd
from huggingface_hub import hf_hub_download
import mlx.core as mx
from mlx_lm import load, generate
from config import *

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model management for MLX-based local models"""
    
    def __init__(self):
        self.models_dir = BASE_DIR / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model instances
        self.text_model = None
        self.text_tokenizer = None
        self.vision_model = None
        
        # Model configurations
        self.text_model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        self.vision_model_name = "mlx-community/Qwen2.5-VL-72B-Instruct-4bit"
        
        # Model states
        self.text_model_loaded = False
        self.vision_model_loaded = False
        
        logger.info("ModelManager initialized")
    
    def download_model(self, model_name: str) -> bool:
        """Download model if not already cached"""
        try:
            logger.info(f"Checking/downloading model: {model_name}")
            
            # MLX-LM automatically handles model downloading and caching
            # Just attempt to load to trigger download if needed
            model_path = self.models_dir / model_name.replace("/", "_")
            
            if not model_path.exists():
                logger.info(f"Model {model_name} not found locally, will download on first load")
            else:
                logger.info(f"Model {model_name} found in cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
    
    def load_text_model(self) -> bool:
        """Load Llama 3.2 3B text analysis model"""
        try:
            if self.text_model_loaded:
                logger.info("Text model already loaded")
                return True
            
            logger.info(f"Loading text model: {self.text_model_name}")
            
            # Load model and tokenizer using MLX-LM
            self.text_model, self.text_tokenizer = load(self.text_model_name)
            
            # Warm-up the model with a simple prompt
            warmup_prompt = "Hello, this is a test."
            _ = generate(self.text_model, self.text_tokenizer, prompt=warmup_prompt, max_tokens=10, verbose=False)
            
            self.text_model_loaded = True
            logger.info("Text model loaded and warmed up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            self.text_model_loaded = False
            return False
    
    def load_vision_model(self) -> bool:
        """Load Sparrow VLM for image analysis (placeholder for now)"""
        try:
            if self.vision_model_loaded:
                logger.info("Vision model already loaded")
                return True
            
            logger.info("Vision model loading will be implemented with Sparrow VLM integration")
            
            # For now, mark as loaded to prevent repeated attempts
            self.vision_model_loaded = True
            logger.info("Vision model placeholder loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vision model: {e}")
            self.vision_model_loaded = False
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate text using loaded Llama model"""
        try:
            if not self.text_model_loaded:
                logger.info("Text model not loaded, attempting to load...")
                if not self.load_text_model():
                    raise Exception("Failed to load text model")
            
            # Generate response
            response = generate(
                self.text_model, 
                self.text_tokenizer, 
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def analyze_text_for_anti_india(self, text: str) -> Dict[str, Any]:
        """Analyze text for anti-India content using Llama 3.2 3B"""
        try:
            # Create the analysis prompt
            prompt = f"""You are an OSINT AI classifier. 
Your task is to analyze a given TEXT and decide if it contains content related to an anti-India campaign.

Instructions:
1. Decide whether the text promotes, supports, or is connected to an anti-India campaign.
2. Give a confidence score between 0 (not related at all) and 1 (definitely related).
3. Explain briefly why you decided that.

Output strictly in JSON format only:
{{
  "is_anti_india": <true/false>,
  "confidence": <float between 0 and 1>,
  "reasoning": "<short reasoning>"
}}

TEXT TO ANALYZE:
"{text}"

JSON Response:"""

            # Generate response
            response = self.generate_text(prompt, max_tokens=200, temperature=0.1)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # Fallback if no JSON found
                return {
                    "is_anti_india": False,
                    "confidence": 0.0,
                    "reasoning": "Could not parse model response"
                }
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate result structure
            if not all(key in result for key in ["is_anti_india", "confidence", "reasoning"]):
                raise ValueError("Invalid response structure")
            
            # Ensure confidence is between 0 and 1 and not NaN
            confidence = result.get("confidence", 0.0)
            try:
                confidence = float(confidence)
                if pd.isna(confidence) or confidence != confidence:  # Check for NaN
                    confidence = 0.0
                result["confidence"] = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                result["confidence"] = 0.0
            
            # Ensure reasoning is a string
            if not isinstance(result.get("reasoning"), str):
                result["reasoning"] = "Analysis completed"
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "is_anti_india": False,
                "confidence": 0.0,
                "reasoning": f"Analysis error: {str(e)}"
            }
    
    def analyze_image_for_anti_india(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for anti-India content (placeholder for Sparrow VLM)"""
        try:
            logger.info(f"Image analysis requested for: {image_path}")
            
            # Placeholder implementation
            # TODO: Integrate with Sparrow VLM
            return {
                "is_anti_india": False,
                "confidence": 0.0,
                "reasoning": "Image analysis with Sparrow VLM not yet implemented"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "is_anti_india": False,
                "confidence": 0.0,
                "reasoning": f"Image analysis error: {str(e)}"
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        return {
            "text_model": {
                "name": self.text_model_name,
                "loaded": self.text_model_loaded,
                "ready": self.text_model is not None
            },
            "vision_model": {
                "name": self.vision_model_name,
                "loaded": self.vision_model_loaded,
                "ready": self.vision_model is not None
            },
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            # Get MLX memory info if available
            memory_info = {
                "mlx_peak_memory": 0,
                "mlx_memory_in_use": 0
            }
            
            # Try to get MLX memory stats
            try:
                memory_info["mlx_peak_memory"] = mx.metal.get_peak_memory() / (1024 * 1024)  # MB
                memory_info["mlx_memory_in_use"] = mx.metal.get_memory_in_use() / (1024 * 1024)  # MB
            except:
                pass
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup models and free memory"""
        try:
            logger.info("Cleaning up models...")
            
            self.text_model = None
            self.text_tokenizer = None
            self.vision_model = None
            
            self.text_model_loaded = False
            self.vision_model_loaded = False
            
            # Clear MLX cache if available
            try:
                mx.metal.clear_cache()
            except:
                pass
            
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

if __name__ == "__main__":
    # Test the model manager
    manager = get_model_manager()
    
    # Test text model loading
    print("Testing text model loading...")
    if manager.load_text_model():
        print("✅ Text model loaded successfully")
        
        # Test text analysis
        test_text = "This is a test message about current events."
        result = manager.analyze_text_for_anti_india(test_text)
        print(f"Analysis result: {result}")
    else:
        print("❌ Failed to load text model")
    
    # Print model status
    status = manager.get_model_status()
    print(f"Model status: {json.dumps(status, indent=2)}")