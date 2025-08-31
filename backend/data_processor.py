import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
import uuid
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from config import *

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def _save_single_split(args):
    """Helper function for parallel split saving"""
    split_df, filename, file_path = args
    try:
        split_df.to_csv(file_path, index=False)
        return str(file_path), len(split_df)
    except Exception as e:
        logger.error(f"Error saving split {filename}: {e}")
        raise

class DataProcessor:
    def __init__(self):
        self.metadata_file = SPLITS_DIR / "metadata.json"
        self.splits_metadata = self.load_metadata()
    
    def load_metadata(self) -> dict:
        """Load existing metadata or create empty dict"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        return {}
    
    def save_metadata(self):
        """Save metadata to file"""
        try:
            SPLITS_DIR.mkdir(exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.splits_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV or XLSX file and return DataFrame"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .xlsx, .xls")
            
            logger.info(f"Loaded {file_extension} file with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def shuffle_data(self, df: pd.DataFrame, random_seed: int = None) -> pd.DataFrame:
        """Shuffle DataFrame rows randomly"""
        if random_seed:
            np.random.seed(random_seed)
        
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        logger.info(f"Shuffled {len(shuffled_df)} rows")
        return shuffled_df
    
    def split_data(self, df: pd.DataFrame, num_splits: int = DEFAULT_NUM_SPLITS) -> List[pd.DataFrame]:
        """Split DataFrame into random chunks ensuring no overlap"""
        total_rows = len(df)
        
        if total_rows < num_splits * MIN_ROWS_PER_SPLIT:
            logger.warning(f"Not enough rows for {num_splits} splits with minimum {MIN_ROWS_PER_SPLIT} rows each")
            num_splits = max(1, total_rows // MIN_ROWS_PER_SPLIT)
        
        # Calculate split sizes
        base_size = total_rows // num_splits
        remainder = total_rows % num_splits
        
        split_sizes = [base_size] * num_splits
        for i in range(remainder):
            split_sizes[i] += 1
        
        # Create splits
        splits = []
        start_idx = 0
        
        for i, size in enumerate(split_sizes):
            end_idx = start_idx + size
            split_df = df.iloc[start_idx:end_idx].copy()
            splits.append(split_df)
            start_idx = end_idx
            
            logger.info(f"Split {i+1}: {len(split_df)} rows")
        
        return splits
    
    def save_splits_parallel(self, splits: List[pd.DataFrame], base_name: str) -> Tuple[List[str], str]:
        """Save split DataFrames to CSV files in parallel"""
        SPLITS_DIR.mkdir(exist_ok=True)
        
        dataset_id = str(uuid.uuid4())[:8]  # Short unique ID for this dataset
        
        # Prepare arguments for parallel processing
        save_args = []
        for i, split_df in enumerate(splits):
            filename = f"{base_name}_{dataset_id}_split_{i+1}.csv"
            file_path = SPLITS_DIR / filename
            save_args.append((split_df, filename, file_path))
        
        split_files = []
        total_rows = 0
        
        # Use ThreadPoolExecutor for I/O bound file operations
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_THREADING) as executor:
            future_to_args = {executor.submit(_save_single_split, args): args for args in save_args}
            
            for future in as_completed(future_to_args):
                try:
                    file_path, rows = future.result()
                    split_files.append(file_path)
                    total_rows += rows
                    logger.info(f"Saved split to {Path(file_path).name}")
                except Exception as e:
                    logger.error(f"Split saving failed: {e}")
        
        # Sort files to maintain order
        split_files.sort()
        
        # Store metadata
        self.splits_metadata[dataset_id] = {
            'base_name': base_name,
            'num_splits': len(splits),
            'split_files': split_files,
            'analyzed_files': [],
            'total_rows': total_rows,
            'served_splits': [],  # Track which splits have been served
            'status': 'processing'  # processing, analyzed, completed
        }
        
        self.save_metadata()
        logger.info(f"Parallel save complete: {len(split_files)} files saved")
        return split_files, dataset_id
    
    def save_splits(self, splits: List[pd.DataFrame], base_name: str) -> Tuple[List[str], str]:
        """Save split DataFrames to CSV files (with parallel processing for large datasets)"""
        if len(splits) >= 3:  # Use parallel processing for 3+ splits
            return self.save_splits_parallel(splits, base_name)
        else:
            # Use sequential processing for small datasets
            SPLITS_DIR.mkdir(exist_ok=True)
            
            split_files = []
            dataset_id = str(uuid.uuid4())[:8]
            
            for i, split_df in enumerate(splits):
                filename = f"{base_name}_{dataset_id}_split_{i+1}.csv"
                file_path = SPLITS_DIR / filename
                
                split_df.to_csv(file_path, index=False)
                split_files.append(str(file_path))
                
                logger.info(f"Saved split {i+1} to {filename}")
            
            # Store metadata
            self.splits_metadata[dataset_id] = {
                'base_name': base_name,
                'num_splits': len(splits),
                'split_files': split_files,
                'analyzed_files': [],
                'total_rows': sum(len(split) for split in splits),
                'served_splits': [],
                'status': 'processing'
            }
            
            self.save_metadata()
            return split_files, dataset_id
    
    def process_file(self, input_file: str, num_splits: int = DEFAULT_NUM_SPLITS, random_seed: int = None) -> Tuple[List[str], str]:
        """Complete pipeline: load, shuffle, split, and save"""
        logger.info(f"Starting processing of {input_file}")
        
        # Extract base name from input file
        base_name = Path(input_file).stem
        
        # Load and process data
        df = self.load_file(input_file)
        
        # Validate required fields exist
        required_fields = [IMAGE_URL_FIELD, DESCRIPTION_FIELD]
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        shuffled_df = self.shuffle_data(df, random_seed)
        splits = self.split_data(shuffled_df, num_splits)
        split_files, dataset_id = self.save_splits(splits, base_name)
        
        logger.info(f"Processing complete. Dataset ID: {dataset_id}")
        return split_files, dataset_id

if __name__ == "__main__":
    processor = DataProcessor()
    # Example usage
    # split_files, dataset_id = processor.process_file("data/input/sample.csv", num_splits=3)
    # split_files, dataset_id = processor.process_file("data/input/sample.xlsx", num_splits=3)
    # print(f"Created {len(split_files)} splits with dataset ID: {dataset_id}")