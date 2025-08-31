import json
import logging
from typing import Dict, Any
from model_manager import get_model_manager

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def text_analysis_llm_block(query: str) -> Dict[str, Any]:
    """
    Analyze text using local MLX Llama 3.2 3B model for anti-India content detection.
    
    Args:
        query: Text to analyze
        
    Returns:
        Dict containing is_anti_india (bool), confidence (float), reasoning (str)
    """
    try:
        # Get model manager instance
        manager = get_model_manager()
        
        # Analyze text using the centralized model manager
        result = manager.analyze_text_for_anti_india(query)
        
        logger.info(f"Text analysis complete. Result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return {
            "is_anti_india": False,
            "confidence": 0.0,
            "reasoning": f"Analysis failed: {str(e)}"
        }


if __name__ == "__main__":
    # Test the text analysis with sample text
    test_text = """
    Introduction
    India and Maldives have shared diplomatic, defence, economic, and cultural relations for the past six decades. Located in a crucial geographical position in the Indian Ocean, Maldives is vital to India's strategy for the Indian Ocean and its neighbourhood. For its part, Maldives reaps benefits from India's economic assistance and net security provision. India has assisted Maldives in various ways since its independence in 1965, such as its pursuit of socio-economic development and modernisation, as well as maritime security. Their engagements flourished beginning in the late 1980s, when India launched 'Operation Cactus' to abort a coup in Maldives against Maumoon Abdul Gayoom's autocratic regime.

    The cordial friendship continued throughout Maldives' first democratic government, elected in 2008.  However, with Abdulla Yameen coming to power in 2013, India-Maldives relations spiralled downward with his crackdown on democracy, proximity towards China, and anti-India rhetoric used to muster nationalist sentiments.  In 2018 a new president, Ibrahim Solih, was elected, who immediately worked to improve the relationship by initiating an 'India First' policy. The policy prioritised India for economic and defence partnerships, and showed greater sensitivity to Indian concerns emanating from Chinese investments and activities in Maldives.

    Not everyone agrees with this policy, however. In October 2020, the opposition coalition—i.e., the Progressive Party of Maldives (PPM) and the People's National Congress (PNC)— officially launched a challenge to the bilateral relationship through what it called the 'India Out' campaign. The campaign seeks to exploit anti-India sentiments, already prevalent parallel to the democratic transition and amidst allegations of India's expansionist ambitions. 'India Out' aims to fuel more hatred by creating scepticism for India's investments in Maldives, the defence partnerships between the two, and India's net-security provisions. Both of the political parties behind the campaign are led by Yameen.

    This paper traces the origins of 'India Out', its nature, and drivers, and its implications.
    """
    
    print("Testing MLX-based text analysis...")
    result = text_analysis_llm_block(test_text)
    print(f"Analysis result: {json.dumps(result, indent=2)}")
