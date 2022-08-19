import base64
import click
import datetime
import requests
import pickle

import pika

IMAGE_PATH = "../../peekingduck/data/input/shiba_inu.jpeg"
OBJECT_NAME = "shiba_inu"
# RabbitMQ related
USERNAME = "peekingduck"
PASSWORD = "admin"
QUEUE_NAME = "task_queue"
EXCHANGE_NAME = "cameras"
EXCHANGE_TYPE = "fanout"


@click.command()
@click.option(
    "--mode", default="req-res", help="Choose between 'req-res', 'queue' or 'pub-sub'"
)
@click.option("--ip_add", default="127.0.0.1", help="IP address to send to.")
@click.option("--port", default=5000, type=int)
def send(mode, ip_add, port):
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%y%m%d-%H%M%S-%f")
    image_encoded = encode_image(IMAGE_PATH)
    data = {"name": OBJECT_NAME, "image": image_encoded, "timestamp": time_str}

    if mode == "req-res":
        url = "http://" + ip_add + ":" + str(port)
        requests.post(url, json=data)
    else:
        credentials = pika.PlainCredentials(username=USERNAME, password=PASSWORD)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=ip_add, credentials=credentials)
        )
        channel = connection.channel()
        data = pickle.dumps(data)
        if mode == "queue":
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=QUEUE_NAME,
                body=data,
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
                ),
            )
            connection.close()
        elif mode == "pub-sub":
            channel.exchange_declare(
                exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE
            )
            channel.basic_publish(exchange=EXCHANGE_NAME, routing_key="", body=data)
            connection.close()
        else:
            raise ValueError(
                "Incorrect mode selected. Only 'req-res', 'queue' or 'pub-sub' permitted."
            )


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        # b64encode() encodes into a bytes-like object
        # .decode("utf-8") is a string method (not base64) that converts it to an ASCII string
        # It removes the prefix "b" that would otherwise appear like b"your_string"
        # On the other side, you'll need to encode("utf-8") to change back to bytes object.
        image_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return image_encoded


if __name__ == "__main__":
    send()
