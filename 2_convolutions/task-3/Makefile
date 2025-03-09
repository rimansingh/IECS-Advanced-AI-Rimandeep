run:
	python main.py

clean:
	rm -rf logs/* results/* saved_models/*

results:
	python scripts/analyze_results.py

setup:
	pip install -r requirements.txt
	mkdir -p logs results saved_models

# Usage:
# make run        - Train model based on config
# make clean      - Clean logs, results, and saved models
# make results    - Analyze experiment results
# make setup      - Install dependencies and prepare folders
