from src.core.errors.exception_handler import AppException
from src.core.logging.logger import setup_logging, get_logger
from src.pipeline.train_pipeline import TrainingPipeline

setup_logging(app_name="recsys", level="DEBUG")
logger = get_logger(__name__)

def main():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.start_training_pipeline()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise AppException(e)
    

if __name__ == "__main__":
    main()