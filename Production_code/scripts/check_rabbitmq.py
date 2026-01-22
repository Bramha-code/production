import pika
import os

RABBITMQ_USER = os.getenv("RABBITMQ_USER", "emc")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "changeme")
RABBITMQ_HOST = "rabbitmq"

try:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            credentials=pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD),
        )
    )
    channel = connection.channel()
    print("Successfully connected to RabbitMQ!")
    connection.close()
except Exception as e:
    print(f"Failed to connect to RabbitMQ: {e}")
