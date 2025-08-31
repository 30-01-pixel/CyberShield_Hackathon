from fastapi import FastAPI, HTTPException
import pandas as pd
import json
import logging
import ast
import re
from datetime import datetime
from typing import Dict, Any, List
from config import *

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI(title="CSV Analysis API", version="1.0.0")

class HashtagExtractor:
    """Utility class for extracting and logging hashtags"""
    
    @staticmethod
    def extract_hashtags_from_string(hashtag_str: str) -> List[str]:
        """Extract hashtags from various string formats"""
        if not hashtag_str or pd.isna(hashtag_str) or hashtag_str == "":
            return []
        
        try:
            # Handle string representation of lists like "['#tag1', '#tag2']"
            if hashtag_str.startswith('[') and hashtag_str.endswith(']'):
                # Use ast.literal_eval for safe evaluation
                hashtag_list = ast.literal_eval(hashtag_str)
                if isinstance(hashtag_list, list):
                    return [tag.strip() for tag in hashtag_list if tag and tag.strip()]
            
            # Handle comma-separated hashtags
            if ',' in hashtag_str:
                hashtags = [tag.strip() for tag in hashtag_str.split(',')]
                return [tag for tag in hashtags if tag and tag.strip()]
            
            # Handle space-separated hashtags
            hashtags = re.findall(r'#\w+', hashtag_str)
            if hashtags:
                return hashtags
            
            # Single hashtag or other format
            if hashtag_str.startswith('#'):
                return [hashtag_str.strip()]
            
            return []
            
        except Exception as e:
            logger.warning(f"Could not parse hashtags from '{hashtag_str}': {e}")
            return []
    
    @staticmethod
    def log_hashtags_to_csv(dataset_id: str, split_number: int, data: List[Dict], hashtags_data: List[Dict]):
        """Log extracted hashtags to CSV file"""
        if not TRACK_HASHTAG_USAGE or not hashtags_data:
            return
        
        try:
            # Ensure hashtags directory exists
            HASHTAGS_DIR.mkdir(exist_ok=True)
            
            # CSV file path
            csv_file = HASHTAGS_DIR / f"hashtags_{dataset_id}.csv"
            
            # Create DataFrame from hashtags data
            df_hashtags = pd.DataFrame(hashtags_data)
            
            # Check if file exists to determine if we need headers
            file_exists = csv_file.exists()
            
            # Append to CSV
            df_hashtags.to_csv(
                csv_file, 
                mode='a' if file_exists else 'w',
                header=not file_exists,
                index=False
            )
            
            logger.info(f"Logged {len(hashtags_data)} hashtag entries to {csv_file}")
            
        except Exception as e:
            logger.error(f"Error logging hashtags to CSV: {e}")
    
    @staticmethod
    def extract_hashtags_from_data(dataset_id: str, split_number: int, data: List[Dict]) -> List[Dict]:
        """Extract hashtags from API response data and prepare for logging"""
        hashtags_data = []
        timestamp = datetime.now().isoformat()
        
        try:
            for row_idx, row in enumerate(data):
                # Extract hashtags from the hashtags field
                hashtag_str = row.get(HASHTAG_FIELD, "")
                hashtags = HashtagExtractor.extract_hashtags_from_string(hashtag_str)
                
                # Create entries for each hashtag
                for hashtag in hashtags:
                    hashtag_entry = {
                        "timestamp": timestamp,
                        "dataset_id": dataset_id,
                        "split_number": split_number,
                        "hashtag": hashtag,
                        "row_id": row_idx,
                        "username": row.get("username", ""),
                        "platform": row.get("platform", ""),
                        "language": row.get("language", ""),
                        "text_is_anti_india": row.get("text_is_anti_india", False),
                        "text_confidence": row.get("text_confidence", 0.0)
                    }
                    hashtags_data.append(hashtag_entry)
            
            return hashtags_data
            
        except Exception as e:
            logger.error(f"Error extracting hashtags from data: {e}")
            return []

# Initialize hashtag extractor
hashtag_extractor = HashtagExtractor()

class DatasetTracker:
    def __init__(self):
        self.metadata_file = SPLITS_DIR / "metadata.json"
        self.tracker_file = SPLITS_DIR / "api_tracker.json"
        self.served_data = self.load_tracker()
    
    def load_metadata(self) -> dict:
        """Load dataset metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load metadata: {e}")
            return {}
    
    def load_tracker(self) -> dict:
        """Load API serving tracker"""
        try:
            if self.tracker_file.exists():
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load tracker: {e}")
        return {}
    
    def save_tracker(self):
        """Save API serving tracker"""
        try:
            SPLITS_DIR.mkdir(exist_ok=True)
            with open(self.tracker_file, 'w') as f:
                json.dump(self.served_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save tracker: {e}")
    
    def get_next_split(self, dataset_id: str) -> Dict[str, Any]:
        """Get next unserved split for dataset"""
        metadata = self.load_metadata()
        
        if dataset_id not in metadata:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        dataset_info = metadata[dataset_id]
        
        # Check if dataset is analyzed
        if dataset_info.get('status') != 'analyzed':
            raise HTTPException(status_code=400, detail=f"Dataset {dataset_id} is not yet analyzed")
        
        analyzed_files = dataset_info.get('analyzed_files', [])
        
        if not analyzed_files:
            raise HTTPException(status_code=400, detail=f"No analyzed files found for dataset {dataset_id}")
        
        # Initialize tracker for this dataset if not exists
        if dataset_id not in self.served_data:
            self.served_data[dataset_id] = {
                'served_splits': [],
                'total_splits': len(analyzed_files)
            }
        
        served_splits = self.served_data[dataset_id]['served_splits']
        
        # Find next unserved split
        for i, file_path in enumerate(analyzed_files):
            if i not in served_splits:
                # Mark this split as served
                self.served_data[dataset_id]['served_splits'].append(i)
                self.save_tracker()
                
                # Load and return the data
                try:
                    df = pd.read_csv(file_path)
                    
                    # Clean up any NaN values that could cause JSON serialization issues
                    df = df.fillna("")  # Replace NaN with empty strings
                    
                    # Convert to dict and ensure JSON serializable values
                    data = df.to_dict('records')
                    
                    # Clean up any remaining problematic values
                    for row in data:
                        for key, value in row.items():
                            if pd.isna(value) or value != value:  # Check for NaN
                                row[key] = None
                            elif isinstance(value, float) and (value == float('inf') or value == float('-inf')):
                                row[key] = None
                    
                    # Extract and log hashtags if tracking is enabled
                    if TRACK_HASHTAG_USAGE:
                        try:
                            hashtags_data = hashtag_extractor.extract_hashtags_from_data(
                                dataset_id, i + 1, data
                            )
                            hashtag_extractor.log_hashtags_to_csv(
                                dataset_id, i + 1, data, hashtags_data
                            )
                            logger.info(f"Extracted and logged {len(hashtags_data)} hashtag entries for dataset {dataset_id}, split {i + 1}")
                        except Exception as e:
                            logger.error(f"Error processing hashtags: {e}")
                    
                    return {
                        "status": "success",
                        "dataset_id": dataset_id,
                        "split_number": i + 1,
                        "total_splits": len(analyzed_files),
                        "remaining_splits": len(analyzed_files) - len(served_splits) - 1,
                        "data": data,
                        "row_count": len(data)
                    }
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    # Remove from served list if loading failed
                    self.served_data[dataset_id]['served_splits'].remove(i)
                    self.save_tracker()
                    raise HTTPException(status_code=500, detail=f"Error loading split data: {str(e)}")
        
        # All splits have been served
        return {
            "status": "completed",
            "message": "No data left - all splits have been served",
            "dataset_id": dataset_id,
            "total_splits_served": len(served_splits),
            "data": []
        }
    
    def reset_dataset_tracker(self, dataset_id: str):
        """Reset serving tracker for a dataset"""
        if dataset_id in self.served_data:
            del self.served_data[dataset_id]
            self.save_tracker()
    
    def get_dataset_status(self, dataset_id: str) -> Dict[str, Any]:
        """Get serving status for a dataset"""
        metadata = self.load_metadata()
        
        if dataset_id not in metadata:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        dataset_info = metadata[dataset_id]
        served_info = self.served_data.get(dataset_id, {'served_splits': [], 'total_splits': 0})
        
        total_splits = len(dataset_info.get('analyzed_files', []))
        served_splits = len(served_info['served_splits'])
        
        return {
            "dataset_id": dataset_id,
            "status": dataset_info.get('status', 'unknown'),
            "total_splits": total_splits,
            "served_splits": served_splits,
            "remaining_splits": total_splits - served_splits,
            "base_name": dataset_info.get('base_name', 'unknown'),
            "total_rows": dataset_info.get('total_rows', 0)
        }

# Initialize tracker
tracker = DatasetTracker()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "CSV Analysis API is running"}

@app.get("/datasets")
async def list_datasets():
    """List all available datasets"""
    try:
        metadata = tracker.load_metadata()
        datasets = []
        
        for dataset_id, info in metadata.items():
            served_info = tracker.served_data.get(dataset_id, {'served_splits': []})
            total_splits = len(info.get('analyzed_files', []))
            served_splits = len(served_info['served_splits'])
            
            datasets.append({
                "dataset_id": dataset_id,
                "base_name": info.get('base_name', 'unknown'),
                "status": info.get('status', 'unknown'),
                "total_splits": total_splits,
                "served_splits": served_splits,
                "remaining_splits": total_splits - served_splits,
                "total_rows": info.get('total_rows', 0)
            })
        
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{dataset_id}")
async def get_data(dataset_id: str):
    """Get next unserved split data for a dataset"""
    try:
        return tracker.get_next_split(dataset_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{dataset_id}")
async def get_dataset_status(dataset_id: str):
    """Get serving status for a specific dataset"""
    try:
        return tracker.get_dataset_status(dataset_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{dataset_id}")
async def reset_dataset(dataset_id: str):
    """Reset serving tracker for a dataset (start serving from beginning)"""
    try:
        tracker.reset_dataset_tracker(dataset_id)
        return {"message": f"Reset serving tracker for dataset {dataset_id}"}
    except Exception as e:
        logger.error(f"Error resetting dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{dataset_id}")
async def get_analysis_stats(dataset_id: str):
    """Get analysis statistics for a dataset"""
    try:
        from analysis_pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline()
        stats = pipeline.get_analysis_stats(dataset_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hashtags/{dataset_id}")
async def get_hashtag_stats(dataset_id: str):
    """Get hashtag usage statistics for a dataset"""
    try:
        hashtags_file = HASHTAGS_DIR / f"hashtags_{dataset_id}.csv"
        
        if not hashtags_file.exists():
            return {
                "dataset_id": dataset_id,
                "status": "no_hashtags",
                "message": "No hashtag data available for this dataset"
            }
        
        # Read hashtag data
        df = pd.read_csv(hashtags_file)
        
        if len(df) == 0:
            return {
                "dataset_id": dataset_id,
                "status": "empty",
                "message": "Hashtag file exists but is empty"
            }
        
        # Calculate statistics
        total_hashtags = len(df)
        unique_hashtags = df['hashtag'].nunique()
        top_hashtags = df['hashtag'].value_counts().head(10).to_dict()
        
        # Anti-India hashtag statistics
        anti_india_df = df[df['text_is_anti_india'] == True]
        anti_india_hashtags = len(anti_india_df)
        top_anti_india_hashtags = anti_india_df['hashtag'].value_counts().head(10).to_dict() if len(anti_india_df) > 0 else {}
        
        # Platform breakdown
        platform_stats = df['platform'].value_counts().to_dict()
        
        # Language breakdown
        language_stats = df['language'].value_counts().to_dict()
        
        return {
            "dataset_id": dataset_id,
            "status": "success",
            "total_hashtag_entries": total_hashtags,
            "unique_hashtags": unique_hashtags,
            "top_hashtags": top_hashtags,
            "anti_india_hashtags": {
                "count": anti_india_hashtags,
                "percentage": round((anti_india_hashtags / total_hashtags) * 100, 2) if total_hashtags > 0 else 0,
                "top_hashtags": top_anti_india_hashtags
            },
            "platform_breakdown": platform_stats,
            "language_breakdown": language_stats,
            "last_updated": df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
        }
        
    except Exception as e:
        logger.error(f"Error getting hashtag stats for {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)