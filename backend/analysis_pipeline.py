import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from config import *
from image_analysis import image_analysis_llm_block
from text_analysis import text_analysis_llm_block

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def _analyze_single_row(args):
    """Helper function for parallel row analysis"""
    idx, row, image_field, description_field, skip_image = args
    results = {}
    
    try:
        # Analyze image if URL exists and not skipped
        if not skip_image and pd.notna(row.get(image_field)):
            img_results = analyze_image_helper(row[image_field])
            results.update(img_results)
        elif skip_image:
            # Add default image analysis values when skipped
            results.update({
                'img_is_anti_india': False,
                'img_confidence': 0.0,
                'img_reasoning': 'Image analysis skipped'
            })
        
        # Analyze text if description exists  
        if pd.notna(row.get(description_field)):
            text_results = analyze_text_helper(row[description_field])
            results.update(text_results)
        
        return idx, results
    except Exception as e:
        logger.error(f"Error analyzing row {idx}: {e}")
        return idx, {
            'img_is_anti_india': False,
            'img_confidence': 0.0,
            'img_reasoning': f'Analysis error: {str(e)}',
            'text_is_anti_india': False,
            'text_confidence': 0.0,
            'text_reasoning': f'Analysis error: {str(e)}'
        }

def analyze_image_helper(image_url: str) -> Dict[str, Any]:
    """Helper function for image analysis"""
    try:
        result = image_analysis_llm_block(image_url)
        return {
            'img_is_anti_india': result.get('is_anti_india', False),
            'img_confidence': result.get('confidence', 0.0),
            'img_reasoning': result.get('reasoning', 'Analysis failed')
        }
    except Exception as e:
        return {
            'img_is_anti_india': False,
            'img_confidence': 0.0,
            'img_reasoning': f'Analysis error: {str(e)}'
        }

def analyze_text_helper(text: str) -> Dict[str, Any]:
    """Helper function for text analysis"""
    try:
        result = text_analysis_llm_block(text)
        return {
            'text_is_anti_india': result.get('is_anti_india', False),
            'text_confidence': result.get('confidence', 0.0),
            'text_reasoning': result.get('reasoning', 'Analysis failed')
        }
    except Exception as e:
        return {
            'text_is_anti_india': False,
            'text_confidence': 0.0,
            'text_reasoning': f'Analysis error: {str(e)}'
        }

class AnalysisPipeline:
    def __init__(self):
        self.metadata_file = SPLITS_DIR / "metadata.json"
    
    def _clean_dataframe_for_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to ensure JSON serialization compatibility"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Replace NaN values with appropriate defaults
        for col in df_clean.columns:
            if col in ['img_confidence', 'text_confidence']:
                df_clean[col] = df_clean[col].fillna(0.0)
                # Replace inf values
                df_clean[col] = df_clean[col].replace([float('inf'), float('-inf')], 0.0)
            elif col in ['img_is_anti_india', 'text_is_anti_india']:
                df_clean[col] = df_clean[col].fillna(False)
            elif col in ['img_reasoning', 'text_reasoning']:
                df_clean[col] = df_clean[col].fillna("Analysis not performed")
            else:
                # For other columns, replace NaN with empty string
                df_clean[col] = df_clean[col].fillna("")
        
        return df_clean
    
    def load_metadata(self) -> dict:
        """Load metadata from file"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load metadata: {e}")
            return {}
    
    def save_metadata(self, metadata: dict):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """Analyze single image URL"""
        try:
            result = image_analysis_llm_block(image_url)
            return {
                'img_is_anti_india': result.get('is_anti_india', False),
                'img_confidence': result.get('confidence', 0.0),
                'img_reasoning': result.get('reasoning', 'Analysis failed')
            }
        except Exception as e:
            logger.error(f"Image analysis failed for {image_url}: {e}")
            return {
                'img_is_anti_india': False,
                'img_confidence': 0.0,
                'img_reasoning': f'Analysis error: {str(e)}'
            }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze single text description"""
        try:
            result = text_analysis_llm_block(text)
            return {
                'text_is_anti_india': result.get('is_anti_india', False),
                'text_confidence': result.get('confidence', 0.0),
                'text_reasoning': result.get('reasoning', 'Analysis failed')
            }
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                'text_is_anti_india': False,
                'text_confidence': 0.0,
                'text_reasoning': f'Analysis error: {str(e)}'
            }
    
    def analyze_split_file_parallel(self, split_file: str, skip_image_analysis: bool = False) -> str:
        """Analyze a single split file with parallel processing"""
        try:
            logger.info(f"Starting parallel analysis of {split_file}")
            df = pd.read_csv(split_file)
            
            # Initialize new columns
            for field in IMG_ANALYSIS_FIELDS + TEXT_ANALYSIS_FIELDS:
                df[field] = None
            
            # Process rows in chunks for better memory management
            chunk_size = min(CHUNK_SIZE_FOR_PARALLEL, len(df))
            total_rows = len(df)
            
            for chunk_start in range(0, total_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_rows)
                chunk_df = df.iloc[chunk_start:chunk_end]
                
                logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size}")
                
                # Prepare arguments for parallel processing
                analysis_args = []
                for idx, row in chunk_df.iterrows():
                    analysis_args.append((idx, row, IMAGE_URL_FIELD, DESCRIPTION_FIELD, skip_image_analysis))
                
                # Use ThreadPoolExecutor for I/O bound API calls (limited to avoid overwhelming APIs)
                with ThreadPoolExecutor(max_workers=min(MAX_WORKERS_THREADING, 2)) as executor:
                    future_to_idx = {executor.submit(_analyze_single_row, args): args[0] for args in analysis_args}
                    
                    for future in as_completed(future_to_idx):
                        try:
                            idx, results = future.result()
                            # Update DataFrame with results
                            for field, value in results.items():
                                if field in df.columns:
                                    df.at[idx, field] = value
                            
                            if (idx - chunk_start + 1) % 10 == 0:  # Log progress every 10 rows
                                logger.info(f"Processed {idx - chunk_start + 1}/{chunk_end - chunk_start} rows in chunk")
                        except Exception as e:
                            logger.error(f"Failed to process row: {e}")
            
            # Clean up any problematic values before saving
            df = self._clean_dataframe_for_json(df)
            
            # Save analyzed file
            analyzed_file = str(split_file).replace('splits', 'analyzed')
            ANALYZED_DIR.mkdir(exist_ok=True)
            df.to_csv(analyzed_file, index=False)
            
            logger.info(f"Parallel analysis complete. Saved to {analyzed_file}")
            return analyzed_file
            
        except Exception as e:
            logger.error(f"Error analyzing {split_file}: {e}")
            raise
    
    def analyze_split_file(self, split_file: str, skip_image_analysis: bool = False) -> str:
        """Analyze a single split file and save results"""
        try:
            df = pd.read_csv(split_file)
            
            # Use parallel processing for larger datasets
            if len(df) >= 20:  # Use parallel for 20+ rows
                return self.analyze_split_file_parallel(split_file, skip_image_analysis)
            
            # Sequential processing for small datasets
            logger.info(f"Starting sequential analysis of {split_file}")
            
            # Initialize new columns
            for field in IMG_ANALYSIS_FIELDS + TEXT_ANALYSIS_FIELDS:
                df[field] = None
            
            # Analyze each row
            for idx, row in df.iterrows():
                logger.info(f"Analyzing row {idx+1}/{len(df)}")
                
                # Analyze image if URL exists and not skipped
                if not skip_image_analysis and pd.notna(row.get(IMAGE_URL_FIELD)):
                    img_results = self.analyze_image(row[IMAGE_URL_FIELD])
                    for field, value in img_results.items():
                        df.at[idx, field] = value
                elif skip_image_analysis:
                    # Add default image analysis values when skipped
                    df.at[idx, 'img_is_anti_india'] = False
                    df.at[idx, 'img_confidence'] = 0.0
                    df.at[idx, 'img_reasoning'] = 'Image analysis skipped'
                
                # Analyze text if description exists
                if pd.notna(row.get(DESCRIPTION_FIELD)):
                    text_results = self.analyze_text(row[DESCRIPTION_FIELD])
                    for field, value in text_results.items():
                        df.at[idx, field] = value
            
            # Clean up any problematic values before saving
            df = self._clean_dataframe_for_json(df)
            
            # Save analyzed file
            analyzed_file = str(split_file).replace('splits', 'analyzed')
            ANALYZED_DIR.mkdir(exist_ok=True)
            df.to_csv(analyzed_file, index=False)
            
            logger.info(f"Sequential analysis complete. Saved to {analyzed_file}")
            return analyzed_file
            
        except Exception as e:
            logger.error(f"Error analyzing {split_file}: {e}")
            raise
    
    def analyze_dataset(self, dataset_id: str, max_workers: int = None, use_parallel: bool = True, skip_image_analysis: bool = False):
        """Analyze all splits for a dataset"""
        if max_workers is None:
            max_workers = MAX_WORKERS_THREADING
        
        metadata = self.load_metadata()
        
        if dataset_id not in metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = metadata[dataset_id]
        split_files = dataset_info['split_files']
        
        logger.info(f"Starting analysis of {len(split_files)} splits for dataset {dataset_id}")
        
        analyzed_files = []
        
        # Process splits sequentially to avoid overwhelming the API
        for split_file in split_files:
            try:
                analyzed_file = self.analyze_split_file(split_file, skip_image_analysis)
                analyzed_files.append(analyzed_file)
            except Exception as e:
                logger.error(f"Failed to analyze {split_file}: {e}")
                # Continue with other files even if one fails
        
        # Update metadata
        metadata[dataset_id]['analyzed_files'] = analyzed_files
        metadata[dataset_id]['status'] = 'analyzed'
        self.save_metadata(metadata)
        
        logger.info(f"Dataset {dataset_id} analysis complete. {len(analyzed_files)} files analyzed.")
        return analyzed_files
    
    def get_analysis_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Get analysis statistics for a dataset"""
        metadata = self.load_metadata()
        
        if dataset_id not in metadata:
            return {"error": "Dataset not found"}
        
        dataset_info = metadata[dataset_id]
        analyzed_files = dataset_info.get('analyzed_files', [])
        
        if not analyzed_files:
            return {"status": "not_analyzed", "analyzed_files": 0}
        
        # Aggregate statistics from all analyzed files
        total_rows = 0
        img_anti_india = 0
        text_anti_india = 0
        avg_img_confidence = 0
        avg_text_confidence = 0
        
        for file_path in analyzed_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                total_rows += len(df)
                
                # Handle boolean columns safely
                img_anti_india += df['img_is_anti_india'].fillna(False).sum()
                text_anti_india += df['text_is_anti_india'].fillna(False).sum()
                
                # Handle confidence columns with NaN protection
                img_conf_mean = df['img_confidence'].fillna(0.0).mean()
                text_conf_mean = df['text_confidence'].fillna(0.0).mean()
                
                # Ensure no NaN values
                avg_img_confidence += 0.0 if pd.isna(img_conf_mean) else img_conf_mean
                avg_text_confidence += 0.0 if pd.isna(text_conf_mean) else text_conf_mean
        
        num_files = len(analyzed_files)
        
        # Calculate safe averages and percentages
        avg_img_conf = (avg_img_confidence / num_files) if num_files > 0 else 0.0
        avg_text_conf = (avg_text_confidence / num_files) if num_files > 0 else 0.0
        img_percentage = (img_anti_india / total_rows * 100) if total_rows > 0 else 0.0
        text_percentage = (text_anti_india / total_rows * 100) if total_rows > 0 else 0.0
        
        # Ensure all values are JSON serializable
        return {
            "status": "analyzed",
            "total_rows": int(total_rows),
            "analyzed_files": int(num_files),
            "img_anti_india_count": int(img_anti_india),
            "text_anti_india_count": int(text_anti_india),
            "avg_img_confidence": round(float(avg_img_conf), 3) if not pd.isna(avg_img_conf) else 0.0,
            "avg_text_confidence": round(float(avg_text_conf), 3) if not pd.isna(avg_text_conf) else 0.0,
            "img_anti_india_percentage": round(float(img_percentage), 2) if not pd.isna(img_percentage) else 0.0,
            "text_anti_india_percentage": round(float(text_percentage), 2) if not pd.isna(text_percentage) else 0.0
        }

if __name__ == "__main__":
    pipeline = AnalysisPipeline()
    # Example usage:
    # pipeline.analyze_dataset("dataset_id_here")