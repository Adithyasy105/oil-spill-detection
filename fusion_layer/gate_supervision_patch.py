"""Standalone self-test for the proposed gate supervision loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gate_supervision_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Supervise the gate toward the more accurate frozen expert."""

    gate = outputs["gate"].clamp(eps, 1.0 - eps)
    p_sar = outputs["p_sar"]
    p_hsi = outputs["p_hsi"]

    err_sar = (p_sar - targets).abs()
    err_hsi = (p_hsi - targets).abs()
    target_gate = torch.sigmoid(8.0 * (err_sar - err_hsi)).detach()

    return F.binary_cross_entropy(gate, target_gate)


def main() -> None:
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    p_sar = torch.tensor([[0.2], [0.7], [0.8], [0.9]])
    p_hsi = torch.tensor([[0.8], [0.95], [0.1], [0.6]])
    gate = torch.full_like(targets, 0.5, requires_grad=True)

    loss = gate_supervision_loss(
        {"gate": gate, "p_sar": p_sar, "p_hsi": p_hsi},
        targets,
    )
    loss.backward()

    expected_target_gate = torch.sigmoid(
        8.0 * ((p_sar - targets).abs() - (p_hsi - targets).abs())
    )

    print("Gate supervision self-test")
    print(f"loss: {loss.item():.6f}")
    print(f"target_gate: {expected_target_gate.view(-1).tolist()}")
    print(f"gradient_present: {gate.grad is not None}")

    assert torch.isfinite(loss)
    assert gate.grad is not None
    assert torch.all((expected_target_gate >= 0.0) & (expected_target_gate <= 1.0))
    print("PASS")


if __name__ == "__main__":
    main()
