"""
Evaluate Qwen LLM coaching responses with MLflow's Guidelines scorer.

Sends running-coach prompts to Qwen (local via Ollama), traces each call,
then uses mlflow.evaluate() with Qwen as both the responder AND the judge.

Prerequisites:
    ollama pull qwen3.5:35b
    pip install ollama mlflow pandas
"""
import os

import mlflow
from mlflow.genai.scorers import Guidelines
import ollama
import pandas as pd

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'llm-tracing'
MODEL_NAME = 'qwen3.5:35b'

COACH_PROMPT = (
    'You are an experienced running coach and data scientist. '
    'Analyze training patterns and give concise, actionable advice. '
    'Keep responses under 200 words.'
)

evals_df = pd.DataFrame([
    {'inputs': {'question': 'What is the fastest shoe to run in?'},
     'expectations': {'answer': 'Carbon plated shoes like the ASICS METASPEED Sky are among the fastest.'}},
    {'inputs': {'question': 'Is elevation a top predictor of speed?'},
     'expectations': {'answer': 'Yes, per Strava analysis elevation is among the top predictors of pace.'}},
    {'inputs': {'question': 'How should I taper before a marathon?'},
     'expectations': {'answer': 'Reduce weekly mileage by 20-30% each week for 2-3 weeks before race day.'}},
    {'inputs': {'question': 'My easy runs feel too hard. What should I change?'},
     'expectations': {'answer': 'Slow down. Easy pace should be conversational, roughly 60-75% of max heart rate.'}},
])


@mlflow.trace
def ask_coach(question):
    """Send a question to the running coach and return the response."""
    response = ollama.chat(
        MODEL_NAME,
        messages=[
            {'role': 'system', 'content': COACH_PROMPT},
            {'role': 'user', 'content': question},
        ],
    )
    return response['message']['content']


scorer = Guidelines(
    name="coaching_quality",
    guidelines=[
        "Response must give actionable advice, not generic platitudes.",
        "Response must reference specific numbers, paces, or percentages when relevant.",
        "Response must be safe and not recommend anything that risks injury.",
        "Response must be under 200 words.",
    ],
)


def main():
    # Route MLflow's judge calls through Ollama's OpenAI-compatible endpoint
    os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
    os.environ['OPENAI_API_KEY'] = 'ollama'
    os.environ['OPENAI_MODEL_NAME'] = MODEL_NAME

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name='qwen_coaching_eval'):
        results = mlflow.genai.evaluate(
            data=evals_df,
            predict_fn=ask_coach,
            scorers=[scorer],
        )
        print(f'\nMetrics: {results.metrics}')

    print(f'\nResults logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')


if __name__ == '__main__':
    main()