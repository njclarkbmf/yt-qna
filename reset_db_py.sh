#!/bin/bash
# One-liner that loads .env values and removes the LanceDB directory
#
python -c "from dotenv import load_dotenv; import os, shutil; load_dotenv(); shutil.rmtree(os.getenv('LANCEDB_PATH', 'lancedb'))"
