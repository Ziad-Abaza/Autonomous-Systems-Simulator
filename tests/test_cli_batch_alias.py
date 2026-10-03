"""`cli batch` is a deprecated alias of `batch-run`.

The old implementation created queued run records nothing consumed;
it now delegates to the canonical BatchScheduler path.
"""
import argparse
from sim_experiment import cli


def test_batch_delegates_to_batch_run(monkeypatch, capsys):
    called = {}

    def fake_run(args):
        called["trainer"] = args.trainer
        called["env_mode"] = args.env_mode
        called["timeout"] = args.timeout
        return 0

    monkeypatch.setattr(cli, "cmd_batch_run", fake_run)
    args = argparse.Namespace(experiment_id="exp_x", seeds=[1, 2],
                              scenarios=None)
    rc = cli.cmd_batch(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "deprecated" in out and "batch-run" in out
    assert called == {"trainer": "ppo", "env_mode": "inprocess",
                      "timeout": 600.0}


def test_batch_subcommand_registered():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    s = sub.add_parser("batch")
    s.add_argument("experiment_id")
    s.add_argument("--seeds", type=int, nargs="*", default=None)
    s.set_defaults(fn=cli.cmd_batch)
    args = parser.parse_args(["batch", "exp_9", "--seeds", "3"])
    assert args.fn is cli.cmd_batch
