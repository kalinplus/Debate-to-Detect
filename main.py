from pathlib import Path
from engine import Debate

# Initialize with model, temperature, and delay between API calls
debate = Debate(model_name="gpt-5.1", T=1, sleep=1)
print("[INFO] debate object inited!")

# Run debate on news text
news_text = "Apple will release a new quantum computer next year."
news_path = Path("sample_news.txt")
debate.run(news_text=news_text, news_path=news_path)