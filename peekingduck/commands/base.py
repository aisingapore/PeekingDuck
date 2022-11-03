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

"""PeekingDuck CLI command base classes."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import click


class AliasedGroup(click.Group):
    """A command group which allows subcommands to be attached. Each of the
    subcommands/subgroups can have aliases.
    """

    def __init__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self._aliases: Dict[str, str] = {}
        self._commands: Dict[str, List[str]] = {}

    def command(self, *args: Any, **kwargs: Dict[str, Any]) -> Callable:
        """A shortcut decorator for declaring and attaching a command to
        the group. This also registers any aliases for the command.
        """
        aliases: List[str] = kwargs.pop("aliases", [])  # type: ignore
        decorator = super().command(*args, **kwargs)
        return self._make_alias(aliases, decorator)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Formats all the commands with their respective aliases (if any) in
        parentheses.
        """
        rows = []
        commands = self.list_commands(ctx)
        max_len = max(len(cmd) for cmd in commands)
        limit = formatter.width - 6 - max_len

        for cmd in commands:
            command = self.get_command(ctx, cmd)
            if command is None or getattr(command, "hidden", False):
                continue
            if cmd in self._commands:
                aliases = ",".join(sorted(self._commands[cmd]))
                cmd = f"{cmd} ({aliases})"
            cmd_help = command.get_short_help_str(limit)
            rows.append((cmd, cmd_help))

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        """Returns the command specified by ``cmd_name`` if present. Resolves
        ``cmd_name`` into its full command name if it is an alias. Accepts a
        prefix for the command so long as it is unique.
        """
        cmd_name = self._resolve_alias(cmd_name)
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        matches = [cmd for cmd in self.list_commands(ctx) if cmd.startswith(cmd_name)]
        if not matches:
            return None
        if len(matches) > 1:
            ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")
        return super().get_command(ctx, matches[0])

    def group(self, *args: Any, **kwargs: Dict[str, Any]) -> Callable:
        """A shortcut decorator for declaring and attaching a group to
        the group. This also registers any aliases for the group.
        """
        aliases: List[str] = kwargs.pop("aliases", [])  # type: ignore
        decorator = super().group(*args, **kwargs)
        return self._make_alias(aliases, decorator)

    def resolve_command(
        self, ctx: click.Context, args: List[Any]
    ) -> Tuple[str, click.Command, List[Any]]:
        """Resolves the full command name."""
        _, cmd, args = super().resolve_command(ctx, args)
        return cmd.name, cmd, args

    def _make_alias(self, aliases: List[str], decorator: Callable) -> Callable:
        """Registers the specified decorator functions to its list of aliases.

        Args:
            aliases (List[str]): List of aliases for decorator.
            decorator (Callable): The decorator function

        Returns:
            (Callable): A decorator function with its aliases stored internally
            if any.
        """
        if not aliases:
            return decorator

        def alias_decorator(func: Callable) -> Callable:
            cmd = decorator(func)
            self._commands[cmd.name] = aliases
            for alias in aliases:
                self._aliases[alias] = cmd.name
            return cmd

        return alias_decorator

    def _resolve_alias(self, cmd_name: str) -> str:
        """Returns the full command name which is mapped to ``cmd_name`` if
        possible. Returns ``cmd_name`` otherwise.
        """
        return self._aliases.get(cmd_name, cmd_name)
