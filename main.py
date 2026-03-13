"""
Debate-to-Detect Main Entry Point
Supports both single text mode and batch processing mode
"""
import argparse
import sys
from pathlib import Path

from engine import Debate
from batch_detect import main as batch_main


def single_mode(args):
    """Run single text detection"""
    print("=" * 60)
    print("Mode: Single Text Detection")
    print("=" * 60)

    # Initialize
    debate = Debate(model_name=args.model, T=args.temperature, sleep=args.sleep)

    # Get text to analyze
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding='utf-8')
    else:
        # Read from stdin
        text = sys.stdin.read()

    if not text or text.strip() == "":
        print("[ERROR] No text provided. Use --text, --file, or pipe from stdin.")
        return

    # Run detection
    output_path = Path(args.output) if args.output else Path("detect_result.txt")
    result = debate.run(text=text, output_path=output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Detection Result:")
    print("=" * 60)
    print(f"Verdict: {result['verdict']}")
    print(f"Detection Score: {result['detection_score']:.3f}")
    print(f"Scores: A={result['scores']['Affirmative']} N={result['scores']['Negative']}")
    print(f"Domain: {result['domain']}")
    print("=" * 60)


def batch_mode(args):
    """Run batch detection"""
    print("=" * 60)
    print("Mode: Batch Detection")
    print("=" * 60)

    # Pass through to batch_detect main function
    # Reconstruct sys.argv for batch_main
    sys.argv = ['batch_detect.py'] + args.batch_args

    # Call batch main
    batch_main()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Debate-to-Detect: AI Text Detection via Multi-Agent Debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text mode
  python main.py --single --text "Your text here"
  python main.py --single --file input.txt

  # Batch mode
  python main.py --batch --data-source test --max-samples 10
  python main.py --batch --data-source main --dataset xsum --source-model gpt4o

  # For more batch options, use:
  python batch_detect.py --help
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', action='store_true',
                           help='Run in single text mode')
    mode_group.add_argument('--batch', action='store_true',
                           help='Run in batch mode')

    # Common parameters
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Model to use (default: gpt-4o-mini)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation (default: 1.0)')
    parser.add_argument('--sleep', type=float, default=1.0,
                       help='Sleep time between API calls in seconds (default: 1.0)')

    # Single mode parameters
    parser.add_argument('--text', type=str,
                       help='Text to analyze (single mode)')
    parser.add_argument('--file', type=str,
                       help='File containing text to analyze (single mode)')
    parser.add_argument('--output', type=str,
                       help='Output file path (single mode, default: detect_result.txt)')

    # Batch mode parameters (passed through to batch_detect.py)
    parser.add_argument('--data-source', type=str, default='test',
                       help='Data source for batch mode (default: test)')
    parser.add_argument('--max-samples', type=int, default=-1,
                       help='Maximum samples for batch mode (default: all)')
    parser.add_argument('--dataset', type=str, default='xsum',
                       help='Dataset for main data source')
    parser.add_argument('--source-model', type=str, default='gpt4o',
                       help='Source model for main data source')
    parser.add_argument('--attack-type', type=str, default='delete',
                       help='Attack type for text_attack data source')
    parser.add_argument('--base-dataset', type=str, default='xsum',
                       help='Dataset for base data source')
    parser.add_argument('--base-source-model', type=str, default='gpt-j-6B',
                       help='Source model for base data source')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')

    return parser.parse_args()


def build_batch_args(args):
    """Build arguments list for batch_detect.py"""
    batch_args = []

    if args.data_source:
        batch_args.extend(['--data-source', args.data_source])
    if args.max_samples > 0:
        batch_args.extend(['--max-samples', str(args.max_samples)])
    if args.dataset:
        batch_args.extend(['--dataset', args.dataset])
    if args.source_model:
        batch_args.extend(['--source-model', args.source_model])
    if args.attack_type:
        batch_args.extend(['--attack-type', args.attack_type])
    if args.base_dataset:
        batch_args.extend(['--base-dataset', args.base_dataset])
    if args.base_source_model:
        batch_args.extend(['--base-source-model', args.base_source_model])
    if args.model:
        batch_args.extend(['--model', args.model])
    if args.temperature != 1.0:
        batch_args.extend(['--temperature', str(args.temperature)])
    if args.sleep != 1.0:
        batch_args.extend(['--sleep', str(args.sleep)])
    if args.threshold != 0.5:
        batch_args.extend(['--threshold', str(args.threshold)])

    return batch_args


def main():
    """Main entry point"""
    args = parse_args()

    # Build batch args if needed
    if args.batch:
        args.batch_args = build_batch_args(args)

    # Run appropriate mode
    if args.single:
        single_mode(args)
    elif args.batch:
        batch_mode(args)


if __name__ == "__main__":
    main()
