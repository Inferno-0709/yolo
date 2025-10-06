# src/main.py
from detection.yolo_pipeline import YOLODetector
from rag.rag_pipeline import RAGPipeline
from utils.helpers import setup_logger, save_json

def main():
    logger = setup_logger()
    logger.info("Starting Visual Insight System...")

    # Initialize components
    detector = YOLODetector()
    rag = RAGPipeline()

    # Input image/video
    image_path = "data/images/sample.jpg"

    # Step 1: Run YOLO detection
    detections = detector.detect_objects(image_path)
    logger.info(f"Detections: {detections}")

    # Step 2: Generate RAG-based insights
    insights = rag.generate_insight(detections)
    logger.info(f"Generated Insight: {insights}")

    # Save outputs
    save_json({"detections": detections, "insight": insights}, "output.json")

if __name__ == "__main__":
    main()
