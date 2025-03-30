import argparse
from pathlib import Path
from text_summary.text_parser import PARSERS
from text_summary.summarize import summarize, ensure_model_exists, SUPPORTED_MODELS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text(file_path: Path) -> str:
    parser = PARSERS.get(file_path.suffix.lower())
    return parser(file_path)

def process_files(model: str, input_dir: str, output_dir: str) -> None:
    ensure_model_exists(model)
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")
    if not input_path.is_dir():
        raise ValueError(f"Input directory '{input_dir}' is not a directory.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_files = set()
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() not in PARSERS:
            continue

        try:
            text = extract_text(file_path)
            summary = summarize(model, text)

            output_file = output_path / (file_path.stem + ".txt")
            output_file.write_text(summary, encoding="utf-8")
            processed_files.add(file_path.name)
            logger.info(f"Processed: {file_path.name} -> {output_file.name}")
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    if not processed_files:
        logger.warning("No files were processed. Check the input directory for supported file types.")
    return processed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize text and PDF files in a directory.")
    parser.add_argument("--input_dir", help="Path to input directory containing the input files")
    parser.add_argument("--output_dir", help="Path to output directory for summary files")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="gemma3:1b",
        help="Model to use for summarization (default: gemma3:1b)",
    )
    args = parser.parse_args()

    process_files(args.model, args.input_dir, args.output_dir)
