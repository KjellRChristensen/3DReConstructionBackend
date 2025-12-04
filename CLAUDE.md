# Claude Code Instructions

Project-specific instructions for Claude Code when working on this codebase.

## Python Version

**ALWAYS use Python 3.x (3.11+ preferred). NEVER use Python 2.7.**

```bash
# Correct
python3 main.py server
python3 -m pytest
python3 scripts/extract_deepcad_subset.py

# WRONG - Never do this
python main.py  # May default to Python 2.7
```

## Virtual Environment

Always activate the virtual environment before running commands:

```bash
source venv/bin/activate
```

## Running the Server

```bash
python3 main.py server --reload --port 7001
```

## Database

- SQLite database at `data/training.db`
- Use `python3 -c "from src.database import init_db; init_db()"` to initialize
- Models table contains VLM configurations including image processor settings

## Training

- Image processor configuration is stored in the `models` database table
- Each model has `image_processor_id` and `image_size` columns
- Trainer reads these from database to avoid image size mismatches
