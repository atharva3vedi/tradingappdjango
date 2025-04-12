import os
import sys
import logging
import schedule
import time
from datetime import datetime
from django.core.management import call_command
from django.core.wsgi import get_wsgi_application
import boto3
from botocore.exceptions import ClientError

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tradingapp.settings')
application = get_wsgi_application()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to S3"""
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        logger.info(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise

def update_models():
    """Update all models and log the results"""
    try:
        logger.info("Starting scheduled model update")
        start_time = datetime.now()
        
        # Update models
        call_command('setup_models')
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Model update completed in {duration}")
        
        # Upload logs to S3 (optional)
        if os.environ.get('AWS_S3_BUCKET'):
            upload_to_s3('model_updates.log', os.environ['AWS_S3_BUCKET'], 
                        f'logs/model_updates_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    except Exception as e:
        logger.error(f"Error during scheduled model update: {str(e)}", exc_info=True)

def main():
    """Main function to run the update scheduler"""
    logger.info("Starting model update scheduler")
    
    # Schedule updates based on environment
    if os.environ.get('CLOUD_ENVIRONMENT'):
        # In cloud environment, run less frequently to save costs
        schedule.every().day.at("00:00").do(update_models)  # Daily update at midnight
        schedule.every().day.at("12:00").do(update_models)  # Additional update at noon
    else:
        # In development, run more frequently for testing
        schedule.every(6).hours.do(update_models)
    
    # Run initial update
    update_models()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Model update scheduler stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in model update scheduler: {str(e)}", exc_info=True)
        sys.exit(1) 