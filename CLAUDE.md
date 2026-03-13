# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Debate-to-Detect (D2D)** is a multi-agent debate framework for AI text detection. The system simulates a structured debate between AI agents arguing that text is AI-generated (Affirmative) versus human-written (Negative), with Judge agents evaluating the debate to deliver a verdict.

**Core Philosophy**: Text authorship becomes clearer through adversarial debate and pattern analysis.

**Key Features**:
- Multi-agent debate system with fixed stances (Affirmative/Negative)
- Domain-aware role profiles for each agent
- Multi-dimensional scoring (Logic, Evidence, Coherence, Style, Depth)
- Configurable debate phases (Opening, Rebuttal, Free, Closing)
- No external evidence system needed (focuses on text patterns)

## Setup and Running

### Prerequisites

**Python Version**: Python 3.9 or later (Python 3.11 recommended)

The code requires these dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install openai>=0.27.0 tiktoken requests backoff
```

### Environment Variables (Required)

The application requires OpenAI API credentials. You can configure them in two ways:

**Option 1: Using .env file (Recommended)**
```bash
# Copy the template and fill in your values
cp .env.template .env

# Edit .env with your actual API credentials
# OPENAI_API_KEY=your-api-key-here
# OPENAI_API_BASE=https://api.openai.com/v1

# Load environment variables from .env (add this to your Python script)
from dotenv import load_dotenv
load_dotenv()
```

**Option 2: Using environment variables**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-endpoint"
```

### Running a Debate
```python
from pathlib import Path
from engine import Debate

# Initialize with model, temperature, and delay between API calls
debate = Debate(model_name="gpt-4o", T=1, sleep=1)

# Run debate on text to analyze
text_to_analyze = "The quantum entanglement principle demonstrates that particles can exhibit correlations that cannot be explained by classical physics."
text_path = Path("sample_text.txt")
debate.run(news_text=text_to_analyze, news_path=text_path)
```

**Parameters**:
- `model_name`: OpenAI model (gpt-4o-mini, gpt-4o, gpt-4, gpt-3.5-turbo)
- `T`: Temperature for generation (0.0-1.0, higher = more creative)
- `sleep`: Delay in seconds between API calls to avoid rate limiting

### Output
Results are automatically saved to `Results/` directory as JSON (default) or text format. Output includes:
- Verdict (AI_GENERATED/HUMAN_WRITTEN/UNCERTAIN)
- Score distribution (Affirmative vs Negative)
- Debate summary with analysis
- Full transcript of all speeches

## Architecture

### System Flow
1. **Domain Detection**: Text domain is classified (e.g., "technology", "politics")
2. **Profile Generation**: Each agent gets a domain-specific professional profile
3. **Debate Phases**: Opening → Rebuttal → Free rounds → Closing
4. **Scoring**: Five dimension-specific judges score each side (0-7, must sum to 7 per judge)
5. **Verdict**: Summary judge generates final conclusion based on scores

### Key Components

**Agent** (`agent.py`): Base LLM agent with memory management
- Handles OpenAI API calls with retry/backoff logic
- Automatic memory summarization when conversation exceeds threshold
- Token-aware response generation

**Debate** (`engine.py`): Main orchestration class
- Initializes all role agents (8 debate roles + 5 scoring judges + 1 summary judge)
- Manages shared conversation context and transcript
- Coordinates debate phases and evidence presentation
- Calculates scores and determines verdict

**EvidenceSystem** (`evidence_system.py`): Evidence retrieval and evaluation
- Extracts keywords from news using LLM
- Searches Wikipedia API for each keyword
- Classifies evidence stance (SUPPORTS_TRUE/SUPPORTS_FALSE/NEUTRAL)
- Filters evidence by stance for each side

**Config** (`config.py`): Centralized configuration
- Model support and API settings
- Debate structure (phases, roles, prompts)
- Scoring dimensions and judge configurations
- Evidence system settings

### Agent Roles

**Debate Agents** (8 total):
- Affirmative/Negative: Opening, Rebuttal, Free, Closing
- Fixed stances: Affirmative always argues text is AI-generated, Negative always argues text is human-written

**Judge Agents** (6 total):
- 5 Scoring Judges: Logic, Evidence, Coherence, Style, Depth
- 1 Summary Judge: Provides final summary and conclusion

## Configuration

### Evidence System (`config.py`)
```python
ENABLE_EVIDENCE = False   # Evidence system disabled for AI text detection
EVIDENCE_PHASE = "Free"   # Phase to present evidence ("Free", "Rebuttal", "Opening")
FREE_ROUNDS = 1           # Number of free debate rounds
```

### Save Settings
```python
AUTO_SAVE = True         # Auto-save after completion
SAVE_DIR = "Results"     # Output directory
SAVE_FMT = "json"        # "json" or "txt"
```

### Token Management
```python
MAX_COMPLETION_TOKENS = 1024        # Max tokens per response
MEMORY_SUMMARIZE_THRESHOLD = 20     # Conversations before summarization
MEMORY_KEEP_RECENT = 10             # Recent conversations to keep after summarizing
```

## Important Implementation Details

**Memory Management**: When shared conversation exceeds `MEMORY_SUMMARIZE_THRESHOLD` (default: 20 turns), the system automatically summarizes older context and keeps only the most recent `MEMORY_KEEP_RECENT` turns.

**Fixed Stances**: Agents have fixed stances enforced through system prompts and stance reminders in user prompts. This prevents agents from abandoning their assigned position during debate.

**Scoring Logic**: Each scoring judge assigns points that sum to exactly 7 between both sides. After all 5 judges, the maximum score is 35. The side with the higher total wins the verdict.

**Domain Context**: The system detects the text domain and generates domain-specific profiles for each debate agent (but not judges), making arguments more relevant and informed.

**AI Text Detection Focus**: Unlike fake news detection, this framework analyzes text patterns, writing style, logical coherence, and depth to determine whether text is AI-generated or human-written. No external evidence retrieval is needed.

## Testing and Debugging

**Viewing Debate Progress**: The Debate class prints each phase and speech to stdout as it happens.

**Saved Results**: Check `Results/` directory for complete debate history including all scoring breakdown and analysis.
