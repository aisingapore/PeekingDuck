# Contributing to PeekingDuck

- [Code of Conduct](#code-of-conduct)
- [Types of Contributions](#types-of-contributions)
  - [Report a Bug](#report-a-bug)
  - [New Nodes and Features](#new-nodes-and-features)
  - [Code Contributions](#code-contributions)
- [Community Discussions](#community-discussions)
- [Code Styles](#code-styles)
  - [Git Commit Messages](#git-commit-messages)
  - [Project Conventions](#project-conventions)
- [Test Suites](#test-suites)

## Code of Conduct

The PeekingDuck team enforces a [code of conduct](CODE_OF_CONDUCT.md) to foster an open and welcoming environment for the community.

## Types of Contributions

We welcome all types of contributions to grow and make PeekingDuck better.

### Report a Bug

Before you report a new bug, do check our [issues tracker](https://github.com/aimakerspace/PeekingDuck/issues) to ensure that the problem hasn't already been reported. If it has, just leave a comment on the existing issue, instead of creating a new issue. If it has not been reported, file a new issue and tag it as a bug. In the issue, please provide a short description of the bug and the steps required to replicate it.

Useful Information 
- Operating System (MacOS/Windows/Linux)
- Python environment (venv/pyenv/conda) and dependencies
- Step by step information to recreate the bug


### New Nodes and Features

If you've created new custom nodes that you'd like to showcase to the community, feel free to do so in our [community discussion channels](#community-discussions). If we find these nodes useful and aligned with PeekingDuck's direction, we'll work with you to incorporate them into the core PeekingDuck package!

If you have suggestions for new **non-node** PeekingDuck features, please open an [issue](https://github.com/aimakerspace/PeekingDuck/issues), describe the feature and why it could be useful to PeekingDuck users.

### Code Contributions

Here is a guide of the general steps to contribute code to PeekingDuck.

- Own an issue from the issues board. This is to prevent duplicate work.

  1. Pick an issue from our [issues tracker](https://github.com/aimakerspace/PeekingDuck/issues) and indicate your interest to work on the issue.
  2. If there are no one else working on the issue, the maintainer will assign the issue to you.
  3. After receiving the confirmation from the maintainer, you may begin work on the issue.

- Contributing code 
  1. Do read our [code styles guidelines](#code-styles).
  2. [Fork](https://docs.github.com/en/github/getting-started-with-github/quickstart/fork-a-repo) the aimakerspace/PeekingDuck repository. For more details in this process, Jake Jarvis has a [useful guide](https://jarv.is/notes/how-to-pull-request-fork-github/) that describes steps 2-6 and 10 in more detail.
  3. Clone the forked branch to your local machine.
  4. \[Recommended\] track the original repository as another remote. After which you will be able to receive updates using `git fetch`. This is useful for long term contributions to the repository.
  5. In your local repository, create a descriptive branch to work on your issue.
  6. Make the required changes within the branch.
  7. Add tests (if possible).
  8. Run the [test suite](#test-suites) and check that it passes.
  9. Push the changes to your github remote.
  10. Send us a pull request to PeekingDuck/dev.
  11. Make changes requested by your reviewer (if any).

Thank you for your contributions!!

## Community Discussions

Join us in our [GitHub discussion board](https://github.com/aimakerspace/PeekingDuck/discussions) or [Discord server](https://discord.gg/7x2VyHVEGX).

. Post your thread regarding

- Questions on the project
- Suggestions/feature requests
- Your custom nodes and projects

## Code Styles

Help us maintain the quality of our project by following the conventions we take below.

### Git Commit Messages

This is a shortened version inspired by [angular's contributing docs](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit-message-header).

A standard git commit message helps us structure the project. 

```bash
<type>(<scope>): <short summary>
  │       │             │
  │       │             └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │       │
  │       └─⫸ Commit Scope: core|cli|changelog|packaging|models|heuristics|cicd-infra|docs-infra|community
  │
  │
  └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|test
```

The `<type>` and `<summary>` fields are required, we encourage filling the `(<scope>)` field where possible.

### Project Conventions

The following conventions are adopted to help maintain code consistency.

- PEP8 convention
- PEP484 type hinting for functions and methods
- Absolute imports instead of relative imports 

## Test Suites

PeekingDuck uses tools like pylint, pytest, mypy and bandit to maintain project quality.
Run these tests locally to ensure that your code passes prior to submission of a Pull Request.

```shell
sh scripts/linter.sh            # pylint for pep8 and code consistency
sh scripts/bandit.sh            # bandit to check for security related issues on dependencies
sh scripts/check_type.sh        # mypy to check for type hints on function/method level
sh scripts/run_tests.sh unit    # pytest for all except model nodes
sh scripts/run_tests.sh mlmodel # pytest for model nodes
sh scripts/usecase_tests.sh     # check standard usecase to ensure it is not broken
```

