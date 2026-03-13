#!/usr/bin/env python3
"""
Test script for AI Text Detection framework

This script tests the adapted Debate-to-Detect framework for AI-generated vs human-written text detection.
"""

from pathlib import Path
from engine import Debate

# Test cases
TEST_CASES = [
    {
        "name": "AI-Generated Text",
        "text": """The principle of quantum entanglement demonstrates that particles can exhibit correlations that cannot be explained by classical physics. When two particles become entangled, the quantum state of each particle cannot be described independently of the other, regardless of the distance separating them. This phenomenon has been experimentally verified through numerous tests, confirming the non-local nature of quantum mechanics. The implications of entanglement extend to quantum computing, quantum cryptography, and our fundamental understanding of reality itself."""
    },
    {
        "name": "Human-Written Text (News)",
        "text": """Breaking: Major tech company announces breakthrough in quantum computing research. Scientists at the institute have developed a new quantum processor that can maintain stable qubits for unprecedented durations. "This is a game-changer," said Dr. Sarah Chen, lead researcher. The team published their findings in Nature yesterday, showing a 300% improvement over previous records. Industry experts are cautiously optimistic, noting that practical applications are still years away. Stock prices rose 8% on the news."""
    },
    {
        "name": "Creative Writing",
        "text": """The morning sun filtered through the curtains, casting dancing shadows across the room. Maria blinked, her eyes adjusting to the light, and stretched lazily. She could smell coffee brewing—dark, rich, inviting. Another day in the city, she thought, swinging her legs out of bed. Her phone buzzed on the nightstand. A text from Jake: "Coffee at 9?" She smiled. Some things never changed."""
    }
]

def main():
    """Run AI text detection tests"""
    print("=" * 80)
    print("AI Text Detection Framework - Testing")
    print("=" * 80)

    # Initialize debate engine
    debate = Debate(model_name="gpt-4o-mini", T=1, sleep=1)

    # Run each test case
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'=' * 80}\n")

        # Create test file path
        test_path = Path(f"test_case_{i}.txt")

        # Run debate
        try:
            debate.run(
                news_text=test_case['text'],
                output_path=test_path
            )
            print(f"\n✓ Test Case {i} completed successfully")
        except Exception as e:
            print(f"\n✗ Test Case {i} failed with error: {e}")

        print("\n" + "=" * 80)

    print("\n✓ All tests completed!")
    print("\nResults saved to: Results/")
    print("\nCheck the output files for:")
    print("  - Verdict (AI_GENERATED / HUMAN_WRITTEN / UNCERTAIN)")
    print("  - Score breakdown by dimension")
    print("  - Full debate transcript")

if __name__ == "__main__":
    main()
