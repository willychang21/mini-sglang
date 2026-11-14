"""Start every process of the server (Chapters 2 & 9).

    main process      : FastAPI / uvicorn API server
    minisgl-TP{i}-scheduler : one scheduler + engine per tensor-parallel rank
    minisgl-tokenizer-{i}   : text -> token ids (N workers)
    minisgl-detokenizer-0   : token ids -> incremental text

The API server waits for one "ready" ack from scheduler rank 0, the
detokenizer and each tokenizer before it starts accepting HTTP requests.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

import torch

from ..utils import exit_when_parent_dies, init_logger
from .args import ServerArgs, parse_args

logger = init_logger(__name__)


def _run_scheduler(args: ServerArgs, rank: int, ack_queue) -> None:
    from ..engine import Engine
    from ..scheduler import Scheduler, ZmqSchedulerIO

    exit_when_parent_dies()
    threads = max(1, (os.cpu_count() or 1) // args.tp_size)
    torch.set_num_threads(threads)
    config = args.engine_config(rank)
    try:
        engine = Engine(config)
        io = ZmqSchedulerIO(
            recv_addr=args.scheduler_addr,
            send_addr=args.detokenizer_addr,
            tp_rank=rank,
            tp_size=args.tp_size,
            tp_cpu_group=engine.tp_cpu_group,
        )
        scheduler = Scheduler(config, io, engine, args.scheduler_config())
    except Exception as e:
        logger.exception("scheduler rank %d failed to start", rank)
        ack_queue.put(f"ERROR: scheduler rank {rank}: {e}")
        raise
    if rank == 0:
        ack_queue.put(f"scheduler (tp={args.tp_size}) ready")
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        pass


def start_backend(args: ServerArgs) -> list:
    from ..tokenizer.server import detokenize_worker, tokenize_worker

    ctx = mp.get_context("spawn")
    ack_queue = ctx.Queue()
    procs = []
    for rank in range(args.tp_size):
        procs.append(
            ctx.Process(
                target=_run_scheduler,
                args=(args, rank, ack_queue),
                daemon=True,
                name=f"minisgl-TP{rank}-scheduler",
            )
        )
    procs.append(
        ctx.Process(
            target=detokenize_worker,
            args=(args.model_path, args.detokenizer_addr, args.frontend_addr, ack_queue, "detokenizer-0"),
            daemon=True,
            name="minisgl-detokenizer-0",
        )
    )
    for i in range(args.num_tokenizers):
        procs.append(
            ctx.Process(
                target=tokenize_worker,
                args=(args.model_path, args.tokenizer_addr(i), args.scheduler_addr, ack_queue, f"tokenizer-{i}"),
                daemon=True,
                name=f"minisgl-tokenizer-{i}",
            )
        )
    for p in procs:
        p.start()
    # 1 scheduler (rank 0) + 1 detokenizer + N tokenizers
    for _ in range(args.num_tokenizers + 2):
        msg = ack_queue.get()
        if msg.startswith("ERROR"):
            for p in procs:
                p.terminate()
            raise RuntimeError(msg)
        logger.info(msg)
    return procs


def launch_server(argv=None) -> None:
    import uvicorn

    from .api_server import FrontendManager, app, set_global_state

    args = parse_args(argv)
    state = FrontendManager(
        model_name=os.path.basename(os.path.normpath(args.model_path)),
        send_addrs=args.tokenizer_addrs,
        recv_addr=args.frontend_addr,
    )
    state.bind()
    set_global_state(state)
    procs = start_backend(args)
    logger.info("API server is ready to serve on http://%s:%d", args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    launch_server(sys.argv[1:])
