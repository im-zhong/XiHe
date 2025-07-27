# 2025/7/26
# zhangzhong

from tests.common import generate_testing_config
from xihe.ckpt import Checkpoint, load_ckpt_from_path
from xihe.defs import defs


def test_checkpoint() -> None:
    checkpoint = Checkpoint(
        config=generate_testing_config(),
        model={"dummy_model_key": "dummy_model_value"},
        optimizer={"dummy_optimizer_key": "dummy_optimizer_value"},
        scheduler={"dummy_scheduler_key": "dummy_scheduler_value"},
        grad_scaler={"dummy_grad_scaler_key": "dummy_grad_scaler_value"},
        step=0,
        dataloader=[{"dummy_dataloader_key": "dummy_dataloader_value"}],
        loss=0.0,
        num_tokens=0,
    )

    checkpoint_path = defs.cache_dir / "test_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"Checkpoint: {checkpoint}")
    checkpoint.save(checkpoint_path)

    loaded_checkpoint = load_ckpt_from_path(checkpoint_path)
    assert loaded_checkpoint is not None, "Loaded checkpoint should not be None"
    print(f"Loaded Checkpoint: {loaded_checkpoint}")
    assert loaded_checkpoint.get_config() == checkpoint.get_config()
    assert loaded_checkpoint.get_model_state_dict() == checkpoint.get_model_state_dict()
    assert (
        loaded_checkpoint.get_optimizer_state_dict()
        == checkpoint.get_optimizer_state_dict()
    )
    assert (
        loaded_checkpoint.get_scheduler_state_dict()
        == checkpoint.get_scheduler_state_dict()
    )
    assert (
        loaded_checkpoint.get_grad_scaler_state_dict()
        == checkpoint.get_grad_scaler_state_dict()
    )
    assert loaded_checkpoint.get_dataloader_state_dict(
        0
    ) == checkpoint.get_dataloader_state_dict(0)
    assert loaded_checkpoint.get_step() == checkpoint.get_step()
    assert loaded_checkpoint.get_state_dict() == checkpoint.get_state_dict()
    assert loaded_checkpoint.get_loss() == checkpoint.get_loss()
    assert loaded_checkpoint.get_num_tokens() == checkpoint.get_num_tokens()
