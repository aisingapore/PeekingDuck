# Contributing to PeekingDuck

- [Code of Conduct](#code-of-conduct)
- [Types of Contribution](#types-of-contribution)
  - [Report a Bug](#report-a-bug)
  - [Community Discussions](#community-discussions)
  - [Code Contributions](#code-contributions)
- [Code Styles](#code-styles)
  - [Git Commit Messages](#git-commit-messages)
  - [Project Conventions](#project-conventions)
- [Test Suites](#test-suites)

## Code of Conduct

Please take a look at our [code of conduct](CODE_OF_CONDUCT.md) and adhere to the rules before contributing.

## Types of Contribution

Help us keep discussions focused by checking if there already exists a thread pertaining to your issue before creating a thread. We reserve the right to reject contributions if there 

### Report a Bug

Create an issue on our [issues tracker](https://github.com/aimakerspace/PeekingDuck/issues) and tag the issue as a bug. In the issue, please provide a short description of the bug and the steps required to replicate it.

Useful Information 
- Operating System   (mac/win/linux)
- Python Environment (venv/pyenv)
- Step by step information to recreate the bug.

### Community Discussions

Join us in our [discussion board](https://github.com/aimakerspace/PeekingDuck/discussions). Post your thread regarding

- Questions on the project
- Suggestions/features requests
- Projects built using our project

### Code Contributions

To contribute to our codebase:

0. !!Before working on the issue, do read our [code styles guidelines](#code-styles)
1. Pick an issue from our [issues tracker](https://github.com/aimakerspace/PeekingDuck/issues) and indicate your interest to work on the issue.
2. After receiving a confirmation from the maintainer, you may begin work on the issue
3. [Fork](https://docs.github.com/en/github/getting-started-with-github/quickstart/fork-a-repo) the aimakerspace/PeekingDuck repository
4. In your forked repository, create a descriptive git branch to work on your issue
5. Make the required changes within the branch
6. Add tests (if possible)
7. Run the [test suite](#test-suites) and check that it passes
8. Push your changes to github
9. Send us a pull request to PeekingDuck/main
10. Make changes as requested by your reviewer (if any)

Thank you for your contribution!!

## Code Styles

Help us maintain the quality of our project by following the conventions we take below.

### Git Commit Messages

This is a shortened version is inspired by [angular's contributing docs](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit-message-header).

A standard git commit message helps us to better structure the project. 

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

The following conventions are adopted to help maintain code consistency within the project.

- PEP8 convention
- PEP484 type hinting for functions and methods
- Absolute imports instead of relative imports 

## Test Suites

The project uses tools like pylint, pytest and bandit to maintain project quality. To run the test in mac. (for linux use `bash` instead of `sh`)

```shell
sh scripts/linter.sh        # pylint for pep8 and code consistency
sh scripts/bandit.sh        # bandit to check for security related issues on dependencies
sh scripts/check_type.sh    # mypy to check for type hints on function/method level
sh scripts/unit_tests.sh    # pytest unit test for individual components
sh scripts/usecase_tests.sh # check standard usecase to ensure it is not broken
```

NOTE - linter, bandit, check_type runs on every pull request
