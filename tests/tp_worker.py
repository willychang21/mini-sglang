"""Helper run in a subprocess per TP rank (spawned by test_tp.py)."""

from __future__ import annotations

import sys

import torch


def run(rank: int, size: int, model: str, port: int, prompt, out_path: str, max_tokens: int) -> None:
    from minisgl import SamplingParams
    from minisgl.config import EngineConfig
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine
    from minisgl.message import UserMsg
    from minisgl.scheduler import OfflineIO, Scheduler
    from minisgl.speculative import SequenceRunner

    torch.set_num_threads(1)
    cfg = EngineConfig(
        model, device="cpu", tp_info=DistributedInfo(rank, size), max_total_tokens=1024,
        max_seq_len=256, dist_init_addr=f"tcp://127.0.0.1:{port}",
    )
    engine = Engine(cfg)
    logits = SequenceRunner(engine, row=1).extend(prompt, all_logits=True)
    io = OfflineIO()
    sched = Scheduler(cfg, io, engine)
    io.put(UserMsg(uid=0, input_ids=prompt, sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True)))
    sched.run_until_idle()
    tokens = [m.next_token for m in io.outbox]
    if rank == 0:
        torch.save({"logits": logits, "tokens": tokens}, out_path)
    engine.shutdown()


if __name__ == "__main__":
    rank, size, model, port, out_path, max_tokens = sys.argv[1:7]
    prompt = [int(x) for x in sys.argv[7].split(",")]
    run(int(rank), int(size), model, int(port), prompt, out_path, int(max_tokens))
