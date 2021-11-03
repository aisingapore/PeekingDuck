import requests
import argparse
import json
import datetime
import pytz

from copy import deepcopy


class StandupBot:
    SHAME_GIF_URL = "https://media3.giphy.com/media/vX9WcCiWwUF7G/200.gif"
    COMMIT_FORMAT = "[#{number}]({url}) - {title} \n "
    DISCORD_JSON = {
        "username": "Stand-up Bot",
        "content": "Hi team, here's a summary of the latest activities :mechanical_arm: :robot:",
        "embeds": [
            {
                "title": None,
                "fields": [
                    {"name": "New Issues since last stand-up", "value": None},
                    {"name": "PRs merged since last stand-up", "value": None},
                    {"name": "PRs opened since last stand-up", "value": None},
                ],
                "footer": {"text": "This message is brought to you by Stand-up Bot"},
            }
        ],
    }

    EMPTY_MESSAGE = {
        "username": "Stand-up Bot",
        "content": "Hi team, seems there wasn't any activity yesterday",
    }

    def __init__(self, opened_issues, merged_pr, opened_pr, webhook_url, date):
        self.opened_issues = opened_issues
        self.merged_pr = merged_pr
        self.opened_pr = opened_pr
        self.discord_message = deepcopy(self.DISCORD_JSON)
        self.DISCORD_WEBHOOK = webhook_url
        self.date = date
        self.is_friday = self._check_friday(self.date)

        self._add_date()
        self.post_to_discord()

    @staticmethod
    def _check_friday(date):
        is_friday = datetime.datetime.strptime(date, "%Y-%m-%d").weekday() == 4
        return is_friday

    @staticmethod
    def _check_weekend(date):
        date_dt = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
        now = pytz.timezone("UTC")
        to_sg = pytz.timezone("Singapore")
        date_converted = now.localize(date_dt).astimezone(to_sg)
        weekday = date_converted.weekday()
        return weekday == 5 or weekday == 6

    def _name_and_shame(self):
        namelist = ", ".join(
            set(
                [
                    "@" + pr["mergedBy"]["login"]
                    for pr in self.merged_pr
                    if self._check_weekend(pr["mergedAt"])
                ]
            )
        )
        return (
            f"Hi {namelist}, you shouldn't really be working over the weekends. "
            "The GIF below represents what I think of you"
        )

    def _add_date(self):
        message_title = "Activity for {}:".format(self.date)
        self.discord_message["embeds"][0]["title"] = message_title

    def add_shame_message(self):
        shame_message = self._name_and_shame()
        self.discord_message["embeds"][0]["image"] = {"url": self.SHAME_GIF_URL}
        self.discord_message["embeds"][0]["fields"].append(
            {"name": "Naming and Shaming Corner", "value": shame_message}
        )

    def parse_json(self, json_to_parse):
        content = ""
        for issue in json_to_parse:
            content += self.COMMIT_FORMAT.format(**issue)
        if not content:
            return "There's nothing here... Hurray?"
        return content

    def post_to_discord(self):
        json_to_post = self.format_to_discord_json()
        print(json_to_post)
        post = requests.post(self.DISCORD_WEBHOOK, json=json_to_post)

    def _check_no_updates(self):
        is_no_updates = False
        if not self.opened_issues and not self.merged_pr and not self.opened_pr:
            is_no_updates = True
        return is_no_updates

    def format_to_discord_json(self):
        if self._check_no_updates():
            return self.EMPTY_MESSAGE

        opened_issues_parsed = self.parse_json(self.opened_issues)
        opened_pr_parsed = self.parse_json(self.opened_pr)
        merged_pr_parsed = self.parse_json(self.merged_pr)

        self.discord_message["embeds"][0]["fields"][0]["value"] = opened_issues_parsed
        self.discord_message["embeds"][0]["fields"][1]["value"] = merged_pr_parsed
        self.discord_message["embeds"][0]["fields"][2]["value"] = opened_pr_parsed

        if self.merged_pr and self.is_friday:
            self.add_shame_message()
        return self.discord_message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("opened_issues")
    parser.add_argument("opened_pr")
    parser.add_argument("merged_pr")
    parser.add_argument("date")
    parser.add_argument("webhook")
    args = parser.parse_args()

    opened_issues = json.loads(args.opened_issues)
    opened_pr = json.loads(args.opened_pr)
    merged_pr = json.loads(args.merged_pr)
    date = args.date
    webhook = args.webhook

    StandupBot(opened_issues, opened_pr, merged_pr, webhook, date)
