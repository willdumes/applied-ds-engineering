import json
from datetime import datetime, timezone
from confluent_kafka import Producer
from growthbook import GrowthBook

producer = Producer({"bootstrap.servers": "localhost:9092"})
TOPIC = "experiment-events"
NUM_USERS = 10_000


def send_exposure(experiment, result, user_context=None, **kwargs):
    """Callback: SDK fires this when a user is bucketed into an experiment."""
    event = {
        "user_id": user_context.attributes["id"],
        "experiment_id": experiment.key,
        "variation_id": result.key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    producer.produce(TOPIC, value=json.dumps(event).encode("utf-8"))


# Simulate NUM_USERS users each evaluating the feature flag.
# The SDK hashes each user's "id" to assign a variation,
# then fires send_exposure() with that user's context.
for i in range(NUM_USERS):
    user_id = f"user-{i}"
    gb = GrowthBook(
        api_host="http://localhost:3100",
        client_key="sdk-YOUR-KEY-HERE",
        on_experiment_viewed=send_exposure,  # ← user_id flows through user_context
        attributes={"id": user_id},
    )
    gb.load_features()
    gb.get_feature_value("my-first-feature", fallback=False)
    gb.destroy()

producer.flush()
print(f"Done — sent exposures for {NUM_USERS} users to '{TOPIC}'")
