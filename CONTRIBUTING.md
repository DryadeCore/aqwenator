# Contributing

Thanks for wanting to help. aqwenator is small on purpose, and most of this page is about
keeping it that way.

## Start with an issue

Ask a question, report a bug, request a feature, or tell us what happened when you ran
aqwenator on your hardware. An issue is the most useful thing you can give this project, and
for most contributions it is all that is needed: maintainers pick it up and implement it,
usually with an agent and with your issue as the specification.

Nothing below is required; all of it helps:

- **What you need and why.** The problem before the solution.
- **What you ran.** Model, GPU, driver, aqwenator version or commit, the command line.
- **What you saw.** Output, numbers, a log. For speed, the replay in
  [docs/benchmarks.md](docs/benchmarks.md) with its run length; for correctness, the last
  line of `scripts/gate.sh` ([docs/correctness.md](docs/correctness.md)).
- **A repro script**, or the exact steps.
- **What you think should change, and where.**
- **What you did not verify.** It saves everyone time.

## Say how it was made

We value transparency. If an AI assistant wrote or helped write an issue or a pull request,
say so, and say which parts you verified yourself: ran it, measured it, read it. It does not
count against the contribution. An unverified number presented as measured does.

## Pull requests

Pull requests are for changes whose approach has been agreed on an issue first. One that
arrives without that is read as an issue: the idea may be taken up, the diff will probably
not be merged, and it will be closed either way. Bug fixes of a few lines are the exception.

What a pull request needs:

- **One intent**, a few hundred lines at most. Larger work is a sequence of pull requests
  that each build, pass the gate and are worth merging alone. If the description needs a
  table of commits, it is several pull requests.
- **Lean.** The shortest change that does the job: no speculative options, no code kept "in
  case". Of two variants, the faster stays and the other is deleted, unless they differ in
  correctness.
- **Correctness first.** Anything touching the engine or the kernels passes
  `scripts/gate.sh` on a GPU; CI cannot run it, so paste its last line. A bug fix comes with
  a check that fails without it.
- **Speed claims are measurements**: before and after, on a named device, with the run
  length, including results that went the wrong way.
- **Not exact means opt-in**, with its quality cost measured against the reference.
- **Sparse comments**: what the code cannot say (a contract, the reason for a decision).
- Nothing specific to one machine in the repository; new dependencies need a reason.
- A short description: what changed, why, the evidence, and what you did not verify.
- [Conventional commit](https://www.conventionalcommits.org) subjects.

## Scope

Qwen models, NVIDIA GPUs, many agents on one device. See [docs/roadmap.md](docs/roadmap.md);
the issue tracker is the place to argue with it.
