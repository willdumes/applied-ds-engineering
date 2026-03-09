import json
import psycopg2
from confluent_kafka import Consumer, KafkaError


KAFKA_CONFIG = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "exposure-writer",
    "auto.offset.reset": "earliest",
}
TOPIC = "experiment-events"
DB_URL = "postgresql://postgres:5432@localhost:5432/ds"


def setup_db(conn):
    """Create the exposures table if it doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exposures (
                user_id       TEXT        NOT NULL,
                experiment_id TEXT        NOT NULL,
                variation_id  TEXT        NOT NULL,
                timestamp     TIMESTAMPTZ NOT NULL
            )
        """)
    conn.commit()


def insert_exposure(conn, event):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO exposures (user_id, experiment_id, variation_id, timestamp) VALUES (%s, %s, %s, %s)",
            (event["user_id"], event["experiment_id"], event["variation_id"], event["timestamp"]),
        )
    conn.commit()


def run():
    conn = psycopg2.connect(DB_URL)
    setup_db(conn)

    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([TOPIC])

    print(f"Listening on '{TOPIC}'...")
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"Kafka error: {msg.error()}")
                break

            event = json.loads(msg.value().decode("utf-8"))
            insert_exposure(conn, event)
            print(f"Wrote: {event['user_id']} → {event['experiment_id']} / {event['variation_id']}")
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    run()
