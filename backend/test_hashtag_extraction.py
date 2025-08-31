#!/usr/bin/env python3

"""
Test script for hashtag extraction functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_server import HashtagExtractor
import json

def test_hashtag_extraction():
    """Test hashtag extraction with various formats"""
    
    test_cases = [
        # String list format (from CSV data)
        "['#FreeKashmir', '#StandWithXYZ']",
        "['#ResistIndia', '#HumanRights']", 
        "['#StopIndia', '#Oppression']",
        "['#BoycottIndia', '#IndiaOut']",
        
        # Single hashtags
        "#FreeKashmir",
        "#IndiaOut",
        
        # Comma-separated
        "#FreeKashmir, #StandWithXYZ",
        "#ResistIndia,#HumanRights",
        
        # Space-separated
        "#FreeKashmir #StandWithXYZ #IndiaOut",
        
        # Empty/null cases
        "",
        None,
        "[]",
        
        # Malformed cases
        "['malformed']",
        "invalid hashtag format",
        "#Test1,#Test2,#Test3"
    ]
    
    print("🧪 Testing Hashtag Extraction")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {repr(test_case)}")
        try:
            result = HashtagExtractor.extract_hashtags_from_string(test_case)
            print(f"  Result: {result}")
            print(f"  Count: {len(result)}")
        except Exception as e:
            print(f"  Error: {e}")

def test_hashtag_data_processing():
    """Test hashtag data extraction from mock API data"""
    
    # Mock data similar to what would be returned by API
    mock_data = [
        {
            "platform": "reddit",
            "username": "testuser1", 
            "hashtags": "['#FreeKashmir', '#StandWithXYZ']",
            "language": "en",
            "text_is_anti_india": True,
            "text_confidence": 0.8
        },
        {
            "platform": "twitter",
            "username": "testuser2",
            "hashtags": "['#ResistIndia', '#HumanRights']", 
            "language": "hi",
            "text_is_anti_india": True,
            "text_confidence": 0.6
        },
        {
            "platform": "twitter",
            "username": "testuser3",
            "hashtags": "",  # Empty hashtags
            "language": "en",
            "text_is_anti_india": False,
            "text_confidence": 0.1
        }
    ]
    
    print("\n\n🔄 Testing Data Processing")
    print("=" * 50)
    
    hashtags_data = HashtagExtractor.extract_hashtags_from_data(
        dataset_id="test123",
        split_number=1,
        data=mock_data
    )
    
    print(f"Processed {len(mock_data)} data rows")
    print(f"Extracted {len(hashtags_data)} hashtag entries")
    
    print("\nHashtag entries:")
    for entry in hashtags_data:
        print(f"  - {entry['hashtag']} (user: {entry['username']}, platform: {entry['platform']}, anti-India: {entry['text_is_anti_india']})")

if __name__ == "__main__":
    test_hashtag_extraction()
    test_hashtag_data_processing()
    print("\n✅ Hashtag extraction tests completed!")