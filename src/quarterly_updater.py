import os
import sys
import logging
import schedule
import time
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [QUARTERLY_UPDATER] - %(levelname)s - %(message)s'
)

def run_pipeline():
    """Executes the data extraction and text preprocessing pipeline."""
    logging.info("Initiating Scheduled Quarterly Dataset Update...")
    
    # Path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_script = os.path.join(base_dir, 'src', 'pipeline.py')

    if not os.path.exists(pipeline_script):
        logging.error(f"Cannot find pipeline script at: {pipeline_script}")
        return

    try:
        # Run pipeline.py as a subprocess
        logging.info("Executing pipeline.py...")
        result = subprocess.run(
            [sys.executable, pipeline_script], 
            capture_output=True, 
            text=True, 
            check=True
        )
        logging.info("Pipeline executed successfully!")
        
        # Display output summary
        lines = result.stdout.split('\n')
        for line in lines[-5:]:  # Print last 5 lines for a summary
            if line.strip():
                logging.info(f"Pipeline StdOut: {line.strip()}")
                
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline execution failed with exit code {e.returncode}.")
        logging.error(f"Error output:\n{e.stderr}")

def main():
    logging.info("Quarterly Updater service started.")
    logging.info("Scheduling pipeline to run every 90 days (Quarterly)...")
    
    # Schedule the job to run every 90 days
    schedule.every(90).days.do(run_pipeline)
    
    # For immediate demonstration/testing purposes, you uncomment the line below:
    # run_pipeline()

    logging.info("Scheduler is active. Standing by...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(3600) # Check scheduler every hour to save CPU cycles
    except KeyboardInterrupt:
        logging.info("Quarterly Updater service manually terminated by user.")

if __name__ == "__main__":
    main()
