# CSV/XLSX Analysis Pipeline

A comprehensive system for processing CSV and XLSX files with image URLs and text descriptions, performing security analysis, and serving results via API with non-repeating data delivery.

## Features

- **Data Processing**: Shuffle and split CSV/XLSX files into random, non-overlapping chunks
- **Security Analysis**: Analyze images and text for anti-India campaign content
- **API Serving**: RESTful endpoints that serve splits sequentially without repetition
- **Progress Tracking**: Track which splits have been served to prevent duplication
- **All Fields Preserved**: Original CSV fields are maintained + analysis results added

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file with your API keys
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

## Usage

### 1. Process CSV/XLSX File
```bash
# Basic processing (shuffle, split, analyze)
python main.py process data/input/your_data.csv
python main.py process data/input/your_data.xlsx

# Custom number of splits
python main.py process data/input/your_data.xlsx --splits 10

# Control parallel processing
python main.py process data/input/your_data.csv --workers 8  # Use 8 parallel workers
python main.py process data/input/your_data.csv --no-parallel  # Disable parallel processing

# Skip analysis (only shuffle and split)
python main.py process data/input/your_data.csv --skip-analysis

# Use random seed for reproducible shuffling
python main.py process data/input/your_data.xlsx --seed 42
```

### 2. Analyze Existing Dataset
```bash
# Analyze splits that were created without analysis
python main.py analyze DATASET_ID

# Control parallel processing for analysis
python main.py analyze DATASET_ID --workers 6  # Use 6 parallel workers
python main.py analyze DATASET_ID --no-parallel  # Disable parallel processing
```

### 3. Start API Server
```bash
# Start server (default: localhost:8000)
python main.py serve

# Custom host and port
python main.py serve --host 0.0.0.0 --port 9000
```

### 4. List Available Datasets
```bash
python main.py list
```

## API Endpoints

### Base URL: `http://localhost:8000`

- **GET /** - Health check
- **GET /datasets** - List all datasets with serving status
- **GET /data/{dataset_id}** - Get next unserved split (non-repeating) + logs hashtags
- **GET /status/{dataset_id}** - Get serving status for dataset
- **GET /stats/{dataset_id}** - Get analysis statistics
- **GET /hashtags/{dataset_id}** - Get hashtag usage statistics and analytics
- **POST /reset/{dataset_id}** - Reset serving tracker (start from beginning)

### API Usage Examples

```bash
# List all datasets
curl http://localhost:8000/datasets

# Get next split (each call returns different split)
curl http://localhost:8000/data/abc12345

# Check how many splits remaining
curl http://localhost:8000/status/abc12345

# Get analysis statistics
curl http://localhost:8000/stats/abc12345

# Get hashtag usage statistics and analytics
curl http://localhost:8000/hashtags/abc12345

# Reset to serve from beginning again
curl -X POST http://localhost:8000/reset/abc12345
```

## Data Flow

1. **Input**: CSV or XLSX file with any number of fields including `image_url` and `description`
2. **Shuffle**: Randomly shuffle all rows for even distribution
3. **Split**: Divide into random, non-overlapping chunks
4. **Analyze**: Process `image_url` and `description` fields using local MLX models
5. **Output**: CSV files with all original fields + 6 new analysis columns
6. **API**: Serve splits sequentially, ensuring no repetition
7. **Hashtag Logging**: Automatically extract and log hashtags to CSV on each API call

## Output Schema

Original CSV fields are preserved, plus these analysis columns:
- `img_is_anti_india` (boolean)
- `img_confidence` (float 0-1)
- `img_reasoning` (string)
- `text_is_anti_india` (boolean) 
- `text_confidence` (float 0-1)
- `text_reasoning` (string)

## Directory Structure

```
├── main.py                 # Main CLI interface
├── config.py              # Configuration settings
├── data_processor.py      # CSV shuffling and splitting
├── analysis_pipeline.py   # Run analysis on split data
├── api_server.py         # FastAPI REST endpoints
├── requirements.txt      # Python dependencies
├── data/
│   ├── input/           # Place your CSV files here
│   ├── splits/          # Generated split CSV files
│   └── analyzed/        # Final analyzed CSV files with results
└── logs/                # Processing logs
```

## Example Workflow

1. Place your CSV or XLSX file in `data/input/`
2. Process: `python main.py process data/input/mydata.csv --splits 5` or `python main.py process data/input/mydata.xlsx --splits 5`
3. Start API: `python main.py serve`
4. Get data: `curl http://localhost:8000/data/{dataset_id}`
5. Each API call returns a different split until all are exhausted
6. When complete: `{"status": "completed", "message": "No data left - all splits have been served"}`

## Non-Repeating Data Guarantee

- Each split contains unique, non-overlapping rows from the original data
- API tracks which splits have been served per dataset
- Sequential serving ensures no data repetition across API calls
- Reset functionality allows restarting from the beginning if needed