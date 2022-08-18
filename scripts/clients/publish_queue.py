"""
Basic image sender as Publisher in PUB/SUB mode using RabbitMQ.
"""
import base64
import datetime
import pickle
import pika

QUEUE = "task_queue"

HOST = "localhost"
# HOST = "http://127.0.0.1:5000/image"
IMAGE_PATH = "../../peekingduck/data/input/shiba_inu.jpeg"
OBJECT_NAME = "shiba_inu"
USERNAME = "peekingduck"
PASSWORD = "admin"

credentials = pika.PlainCredentials(username=USERNAME, password=PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=HOST, credentials=credentials)
)
channel = connection.channel()

channel.queue_declare(queue=QUEUE, durable=True)

# prepare your message body
with open(IMAGE_PATH, "rb") as image_file:
    # b64encode() encodes into a bytes-like object
    # .decode("utf-8") is a string method (not base64) that converts it to an ASCII string
    # It removes the prefix "b" that would otherwise appear like b"your_string"
    # On the other side, you'll need to encode("utf-8") to change back to bytes object.
    image_encoded = base64.b64encode(image_file.read()).decode("utf-8")

current_time = datetime.datetime.now()
time_str = current_time.strftime("%y%m%d_%H%M%S_%f")

data = {"name": OBJECT_NAME, "image": image_encoded, "timestamp": time_str}
data = pickle.dumps(data)

# send the message
channel.basic_publish(
    exchange="",
    routing_key=QUEUE,
    body=data,
    properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE),
)

connection.close()
