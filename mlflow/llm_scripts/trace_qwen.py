"""Trace Qwen LLM calls to MLflow via the Ollama SDK.

Sends running-coach prompts to Qwen (local via Ollama) and logs
full request/response traces to the MLflow Traces tab.

Prerequisites:
    ollama pull qwen3.5:35b
    pip install ollama mlflow
"""

import mlflow
import ollama

MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'llm-tracing'
MODEL_NAME = 'qwen3.5:35b'

COACH_PROMPT = (
    'You are an experienced running coach and data scientist. '
    'Analyze training patterns and give concise, actionable advice. '
    'Keep responses under 200 words.'
)


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


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    questions = [
        'What are the top 3 factors that predict marathon race-day pace?',
        'My model shows carbon race shoes improve pace by 0.3 m/s. Is that plausible or confounding?',
        'I have per-second GPS data for 75 runs. What features would you engineer for a pace prediction model?',
    ]

    with mlflow.start_run(run_name='qwen_coaching_analysis'):
        for question in questions:
            print(f'\nQ: {question}')
            answer = ask_coach(question)
            print(f'A: {answer}')

    print(f'\nTraces logged to {MLFLOW_TRACKING_URI}, experiment: {EXPERIMENT_NAME}')


if __name__ == '__main__':
    main()
