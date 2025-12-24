from dataclasses import dataclass
import torch
from .config import TrainConfig, log

@dataclass
class TrainState:
    global_step: int = 0
    opt_step: int = 0

def train_epochs(
    *,
    cfg: TrainConfig,
    dataset,
    bucket_map,
    step_fn,
    optimizer,
    lr_scheduler,
    trainable_params,
    on_epoch_end=None,
    timer=None
):
    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)
    state = TrainState()

    for epoch in range(1, cfg.epochs + 1):
        for bucket_res, bucket_indices in bucket_map.items():
            log(f"STATUS training bucket_res={bucket_res} samples={len(bucket_indices)}")

            if cfg.shuffle:
                bucket_indices = torch.tensor(bucket_indices)[torch.randperm(len(bucket_indices))].tolist()

            num_samples = len(bucket_indices)
            bs = cfg.batch_size

            for start in range(0, num_samples, bs):
                batch_indices = bucket_indices[start:start + bs]
                loss = step_fn(batch_indices, bucket_res)

                (loss / cfg.grad_accum_steps).backward()
                state.global_step += 1

                if state.global_step % cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state.opt_step += 1

                    eta = timer.update(state.opt_step) if timer else None

                    if state.opt_step % cfg.log_every == 0:
                        lr = (
                            lr_scheduler.get_last_lr()[0]
                            if hasattr(lr_scheduler, "get_last_lr")
                            else optimizer.param_groups[0]["lr"]
                        )

                        msg = (
                            f"TRAIN epoch={epoch} "
                            f"opt_step={state.opt_step} "
                            f"lr={lr:.8f} "
                            f"loss={loss.item():.6f}"
                        )

                        if eta is not None:
                            mins = int(eta // 60)
                            secs = int(eta % 60)
                            msg += f" eta={mins:02d}:{secs:02d}"

                        log(msg)

        if on_epoch_end is not None:
            on_epoch_end(epoch, state)

    return state
