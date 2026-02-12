"""
VocalBaby CLI - Command Line Interface

Provides console entry points:
- vocalbaby-serve: Run the FastAPI prediction server
- vocalbaby-train: Run the full training pipeline
"""

import sys


def serve():
    """
    Run the VocalBaby FastAPI prediction server.
    
    Usage:
        vocalbaby-serve
        
    Or with uvicorn options:
        uvicorn vocalbaby.api.app:app --host 0.0.0.0 --port 8000
    """
    try:
        import uvicorn
        from vocalbaby.logging.logger import logging
        
        logging.info("Starting VocalBaby FastAPI server...")
        uvicorn.run(
            "vocalbaby.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
        )
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


def train():
    """
    Run the VocalBaby training pipeline.
    
    Usage:
        vocalbaby-train
    """
    try:
        from vocalbaby.pipeline.training_pipeline import TrainingPipeline
        from vocalbaby.logging.logger import logging
        
        logging.info("Starting VocalBaby training pipeline...")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        logging.info("Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # For testing purposes
    if len(sys.argv) > 1:
        if sys.argv[1] == "serve":
            serve()
        elif sys.argv[1] == "train":
            train()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Available commands: serve, train")
            sys.exit(1)
    else:
        print("Usage: python -m vocalbaby.cli [serve|train]")
        sys.exit(1)
