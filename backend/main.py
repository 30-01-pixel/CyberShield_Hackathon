#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
import uvicorn
from data_processor import DataProcessor
from analysis_pipeline import AnalysisPipeline
from config import *

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure all required directories exist"""
    directories = [DATA_DIR, INPUT_DIR, SPLITS_DIR, ANALYZED_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)
    logger.info("Directory structure initialized")

def process_file_command(args):
    """Process CSV/XLSX file - shuffle, split, and analyze"""
    try:
        setup_directories()
        
        input_file = Path(args.input_file)
        if not input_file.exists():
            logger.error(f"Input file {input_file} does not exist")
            return False
        
        # Step 1: Process file (shuffle and split)
        logger.info(f"Step 1: Processing {input_file.suffix} file - shuffle and split")
        processor = DataProcessor()
        split_files, dataset_id = processor.process_file(
            str(input_file), 
            num_splits=args.splits,
            random_seed=args.seed
        )
        
        logger.info(f"Created {len(split_files)} splits for dataset {dataset_id}")
        
        # Step 2: Analyze splits
        if not args.skip_analysis:
            logger.info("Step 2: Running analysis pipeline")
            pipeline = AnalysisPipeline()
            
            # Configure parallel processing based on args
            use_parallel = not args.no_parallel
            workers = args.workers if not args.no_parallel else 1
            
            if args.no_parallel:
                logger.info("Parallel processing disabled - using sequential analysis")
            else:
                logger.info(f"Using parallel processing with {workers} workers")
            
            analyzed_files = pipeline.analyze_dataset(dataset_id, max_workers=workers, use_parallel=use_parallel, skip_image_analysis=args.no_image_analysis)
            
            logger.info(f"Analysis complete. {len(analyzed_files)} files analyzed")
            logger.info(f"Dataset {dataset_id} is ready for API serving")
        else:
            logger.info("Skipping analysis (--skip-analysis flag used)")
        
        print(f"\n✅ Processing complete!")
        print(f"Dataset ID: {dataset_id}")
        print(f"Splits created: {len(split_files)}")
        
        if not args.skip_analysis:
            print(f"Analysis complete - ready for API serving")
            print(f"Start API server: python main.py serve")
            print(f"Get data: curl http://localhost:{API_PORT}/data/{dataset_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return False

def analyze_command(args):
    """Analyze existing splits for a dataset"""
    try:
        logger.info(f"Starting analysis for dataset {args.dataset_id}")
        pipeline = AnalysisPipeline()
        
        # Configure parallel processing based on args
        use_parallel = not args.no_parallel
        workers = args.workers if not args.no_parallel else 1
        
        if args.no_parallel:
            logger.info("Parallel processing disabled - using sequential analysis")
        else:
            logger.info(f"Using parallel processing with {workers} workers")
        
        analyzed_files = pipeline.analyze_dataset(args.dataset_id, max_workers=workers, use_parallel=use_parallel, skip_image_analysis=args.no_image_analysis)
        
        logger.info(f"Analysis complete. {len(analyzed_files)} files analyzed")
        print(f"\n✅ Analysis complete for dataset {args.dataset_id}!")
        print(f"Files analyzed: {len(analyzed_files)}")
        print(f"Start API server: python main.py serve")
        
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return False

def serve_command(args):
    """Start the API server"""
    try:
        logger.info(f"Starting API server on {args.host}:{args.port}")
        print(f"🚀 Starting API server...")
        print(f"📍 Server URL: http://{args.host}:{args.port}")
        print(f"📖 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"📋 List datasets: http://{args.host}:{args.port}/datasets")
        print(f"\nPress Ctrl+C to stop the server")
        
        # Import here to avoid circular imports
        from api_server import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

def list_datasets_command(args):
    """List all available datasets"""
    try:
        processor = DataProcessor()
        metadata = processor.splits_metadata or processor.load_metadata()
        
        if not metadata:
            print("No datasets found.")
            return True
        
        print("\n📊 Available Datasets:")
        print("-" * 80)
        
        for dataset_id, info in metadata.items():
            status = info.get('status', 'unknown')
            splits = len(info.get('analyzed_files', info.get('split_files', [])))
            rows = info.get('total_rows', 0)
            name = info.get('base_name', 'unknown')
            
            print(f"Dataset ID: {dataset_id}")
            print(f"  Name: {name}")
            print(f"  Status: {status}")
            print(f"  Splits: {splits}")
            print(f"  Total Rows: {rows}")
            print()
        
        return True
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="CSV Analysis Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process CSV/XLSX file - shuffle, split, and analyze')
    process_parser.add_argument('input_file', help='Path to input CSV or XLSX file')
    process_parser.add_argument('--splits', type=int, default=DEFAULT_NUM_SPLITS, help=f'Number of splits (default: {DEFAULT_NUM_SPLITS})')
    process_parser.add_argument('--seed', type=int, help='Random seed for shuffling')
    process_parser.add_argument('--skip-analysis', action='store_true', help='Skip analysis step (only shuffle and split)')
    process_parser.add_argument('--workers', type=int, default=MAX_WORKERS_THREADING, help=f'Number of parallel workers for analysis (default: {MAX_WORKERS_THREADING})')
    process_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    process_parser.add_argument('--no-image-analysis', action='store_true', help='Skip image analysis (text only)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing splits for a dataset')
    analyze_parser.add_argument('dataset_id', help='Dataset ID to analyze')
    analyze_parser.add_argument('--workers', type=int, default=MAX_WORKERS_THREADING, help=f'Number of parallel workers (default: {MAX_WORKERS_THREADING})')
    analyze_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    analyze_parser.add_argument('--no-image-analysis', action='store_true', help='Skip image analysis (text only)')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the API server')
    serve_parser.add_argument('--host', default=API_HOST, help=f'Host to bind to (default: {API_HOST})')
    serve_parser.add_argument('--port', type=int, default=API_PORT, help=f'Port to bind to (default: {API_PORT})')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available datasets')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    success = False
    if args.command == 'process':
        success = process_file_command(args)
    elif args.command == 'analyze':
        success = analyze_command(args)
    elif args.command == 'serve':
        success = serve_command(args)
    elif args.command == 'list':
        success = list_datasets_command(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())