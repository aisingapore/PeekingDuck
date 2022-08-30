# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to send messages to PeekingDuck Server
"""


import base64
import datetime
import json
import requests

import click
import pika

EXCHANGE_TYPE = "fanout"


host_option = click.option("--host", default="127.0.0.1", help="""IP address of host""")
username_option = click.option(
    "--username", default="peekingduck", help="""Username for RabbitMQ authentication"""
)
password_option = click.option(
    "--password",
    prompt=True,
    hide_input=True,
    help="""Password for RabbitMQ authentication""",
)
image_path_option = click.option("--image_path", help="""Path of image file to send""")
image_descr_option = click.option(
    "--image_descr", default="image", help="""Description of image"""
)


@click.group()
def cli() -> None:
    """Send image and metadata to PeekingDuck Server"""


@click.command()
@host_option
@image_path_option
@image_descr_option
@click.option("--port", default=5000, type=int, help="""Port to send to""")
def req_res(host, image_path, image_descr, port):
    """Request response mode"""
    data = prepare_data(image_path, image_descr)
    url = "http://" + host + ":" + str(port)
    requests.post(url, json=data)


@click.command()
@host_option
@image_path_option
@image_descr_option
@username_option
@password_option
@click.option(
    "--exchange_name", default="pkd_exchange", help="""Name of RabbitMQ exchange"""
)
def pub_sub(host, image_path, image_descr, username, password, exchange_name):
    """Publish subscribe mode"""
    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=host, credentials=credentials)
    )
    channel = connection.channel()
    data = prepare_data(image_path, image_descr)
    body = json.dumps(data)
    channel.exchange_declare(exchange=exchange_name, exchange_type=EXCHANGE_TYPE)
    channel.basic_publish(exchange=exchange_name, routing_key="", body=body)
    connection.close()


@click.command()
@host_option
@image_path_option
@image_descr_option
@username_option
@password_option
@click.option("--queue_name", default="pkd_queue", help="""Name of RabbitMQ queue""")
def queue(host, image_path, image_descr, username, password, queue_name):
    """Message queue mode"""
    credentials = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=host, credentials=credentials)
    )
    channel = connection.channel()
    data = prepare_data(image_path, image_descr)
    body = json.dumps(data)
    channel.queue_declare(queue=queue_name, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=queue_name,
        body=body,
        properties=pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
        ),
    )
    connection.close()


def prepare_data(image_path, image_descr):
    """Prepare image and metadata"""
    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%y%m%d-%H%M%S-%f")
    image_encoded = encode_image(image_path)
    return {"descr": image_descr, "image": image_encoded, "timestamp": time_str}


def encode_image(image_path):
    """Encode image into ASCII string"""
    with open(image_path, "rb") as image_file:
        # b64encode() encodes into a bytes-like object
        # .decode("utf-8") is a string method (not base64) that converts it to an ASCII string
        # It removes the prefix "b" that would otherwise appear like b"your_string"
        # On the other side, you'll need to encode("utf-8") to change back to bytes object.
        image_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return image_encoded


cli.add_command(req_res)
cli.add_command(queue)
cli.add_command(pub_sub)


if __name__ == "__main__":
    cli()
