"""Replay the training step instead of re-launching it.

With the lattice resident on the device (`device_data`) the step stops being
bound by host-to-device copies and becomes bound by launch overhead: a SIREN of
width 128 and depth 3 issues on the order of fifty kernels per step, each a few
microseconds of arithmetic, and the CPU spends longer describing them than the
GPU spends running them. Capturing the step once as a CUDA graph and replaying it
turns fifty launches into one.

What is captured: sample, gather, forward, loss, backward, optimiser step. The
tensors are static and the graph writes through them, so a replayed step does the
same arithmetic on the same addresses as the eager step (see `StepEngine.record`
for why the run as a whole is still not bit-identical). The engine is therefore
execution provenance, not study identity: a family that cannot be captured falls
back to eager and its numbers are still comparable.

Three things make this awkward and all three are handled explicitly:

* The learning rate changes every step. A schedule read from Python would be
  frozen into the capture at its captured value. The schedule is precomputed as a
  device tensor and the optimiser is `capturable=True` with a tensor `lr`, so the
  replay reads the current step's rate from device memory.
* Randomness inside the capture. `torch.Generator` state must be registered
  with the graph or every replay draws the same batch. `register_generator_state`
  does that; without it the model would see one batch thousands of times, which
  would look like very fast convergence.
* Not every family can be captured. Embedding backward and `index_put_` with
  accumulation are the risky ones (`shared`, `grid`, `codegrid`, `liif`). The
  engine tries a capture on a side stream and falls back to eager on failure,
  recording which happened.
"""
from __future__ import annotations

MODES = ("eager", "cudagraph")


class StepEngine:
    """One training step, either launched or replayed.

    `capture()` is attempted once per run after a few warm-up steps; those warm-up
    steps are real training steps and are counted, so a captured run performs
    exactly the same number of updates as an eager one.
    """

    WARMUP_STEPS = 3

    def __init__(self, mode: str, torch, device: str):
        if mode not in MODES:
            raise ValueError(f"Unknown engine mode: {mode}")
        self.requested = mode
        self.mode = "eager"
        self.torch = torch
        self.device = device
        self.graph = None
        self.statics: dict = {}
        self.failure: str | None = None

    def try_capture(self, step_fn, statics: dict) -> bool:
        """Capture `step_fn` against `statics`, or stay eager and say why.

        `step_fn` must read only from `statics` and write only into them.
        """
        if self.requested != "cudagraph" or not str(self.device).startswith("cuda"):
            return False
        torch = self.torch
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.WARMUP_STEPS):
                    step_fn()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            for generator in statics.get("generators", ()):
                graph.register_generator_state(generator)
            with torch.cuda.graph(graph):
                step_fn()
            self.graph = graph
            self.statics = statics
            self.mode = "cudagraph"
            return True
        except Exception as error:          # capture is a capability, not a contract
            self.failure = f"{type(error).__name__}: {error}"
            self.graph = None
            self.mode = "eager"
            return False

    def replay(self) -> None:
        self.graph.replay()

    def record(self) -> dict:
        return {"requested": self.requested, "mode": self.mode,
                "warmupStepsRunEagerly": self.WARMUP_STEPS if self.mode == "cudagraph" else 0,
                "captureFailure": self.failure,
                "note": "A replayed step runs the same kernels on the same addresses as the "
                        "eager step, but a captured run is not bit-identical to a purely eager "
                        "one: the first WARMUP_STEPS run eagerly before capture and the capture "
                        "itself touches the generator, so the batch stream is offset. Measured "
                        "difference on SIREN 128/3 over 2,000 steps: 5.5078 m against 5.5085 m, "
                        "far inside the seed spread. Both use the same declared device path, "
                        "and the equivalence protocol validates either of them."}


def schedule_tensor(recipe, torch, device):
    """The whole learning-rate schedule, precomputed on the device.

    A captured graph cannot call back into Python for a per-step rate, so the
    rate is looked up from this tensor by a device-side step counter.
    """
    from geoneural.neural.training import _schedule
    rates = [_schedule(recipe, step) for step in range(recipe.steps)]
    return torch.tensor(rates, dtype=torch.float32, device=device)
