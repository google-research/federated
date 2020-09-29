workspace(name = "org_federated_research")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.0.2",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()
