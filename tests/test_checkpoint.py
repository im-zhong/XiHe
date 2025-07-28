# 2025/7/26
# zhangzhong
# ruff: noqa: PLR2004

from tests.common import generate_testing_config
from xihe.ckpt import Checkpoint, CheckpointManager, load_ckpt_from_path
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


def test_checkpoint_manager() -> None:
    # config = generate_testing_config()

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

    ckpt_dir = defs.cache_dir / "test_checkpoint_manager"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 为什么，ckpt mgr需要config？
    # 如果需要保存在ckpt中，那么保存的时候从外部传入不久行了吗
    keep_num = 3
    save_steps = 100
    ckpt_mgr = CheckpointManager(
        keep_num=keep_num,
        save_steps=save_steps,
        ckpt_dir=ckpt_dir,
    )
    ckpt_mgr.clear_checkpoints()
    assert ckpt_mgr.get_best_loss() == float("inf"), (
        "Initial best loss should be infinity"
    )

    assert ckpt_mgr.need_save(step=0, loss=100)
    assert ckpt_mgr.need_save_interval(step=0)
    # 在最开始的时候，因为没有保存best loss模型，所以一个相当高的loss也需要保存
    assert ckpt_mgr.need_save_best(loss=100)

    # 咱们要测试什么？
    # 其实就是两个功能啊，一个是按照step inteval来保存
    # 一个就是保存loss最小的
    # 现在开始保存一个checkpoint
    checkpoint = Checkpoint(
        config=generate_testing_config(),
        model={"dummy_model_key": "dummy_model_value"},
        optimizer={"dummy_optimizer_key": "dummy_optimizer_value"},
        scheduler={"dummy_scheduler_key": "dummy_scheduler_value"},
        grad_scaler={"dummy_grad_scaler_key": "dummy_grad_scaler_value"},
        step=0,
        dataloader=[{"dummy_dataloader_key": "dummy_dataloader_value"}],
        loss=100,
        num_tokens=0,
    )
    # 可以看到，他同时保存了两个ckpt
    # 一个是step0 一个是best ckpt
    ckpt_mgr.save_checkpoint(checkpoint)
    assert ckpt_mgr.get_best_loss() == 100.0
    # 咱们需要添加验证啊
    # 直接读取目录，确认目录里面的文件才对
    ckpt_files = list(ckpt_dir.iterdir())
    assert len(ckpt_files) == 2, "There should be two checkpoint files saved."
    assert "ckpt_step_0.tar" in [f.name for f in ckpt_files]
    assert "best_ckpt_step_0.tar" in [f.name for f in ckpt_files]

    # 这个时候再来判断
    assert not ckpt_mgr.need_save(step=1, loss=200)
    assert not ckpt_mgr.need_save_interval(step=1)
    assert not ckpt_mgr.need_save_best(loss=200)

    assert ckpt_mgr.need_save(step=2, loss=50)
    ckpt = Checkpoint(
        config=generate_testing_config(),
        model={"dummy_model_key": "dummy_model_value"},
        optimizer={"dummy_optimizer_key": "dummy_optimizer_value"},
        scheduler={"dummy_scheduler_key": "dummy_scheduler_value"},
        grad_scaler={"dummy_grad_scaler_key": "dummy_grad_scaler_value"},
        step=2,
        dataloader=[{"dummy_dataloader_key": "dummy_dataloader_value"}],
        loss=50,
        num_tokens=0,
    )
    ckpt_mgr.save_checkpoint(ckpt)
    assert ckpt_mgr.get_best_loss() == 50.0

    ckpt_files = list(ckpt_dir.iterdir())
    assert len(ckpt_files) == 2, "There should be two checkpoint files saved."
    assert "ckpt_step_0.tar" in [f.name for f in ckpt_files]
    assert "best_ckpt_step_2.tar" in [f.name for f in ckpt_files]

    # 现在重新构建ckpt mgr对象
    ckpt_mgr = CheckpointManager(
        keep_num=keep_num,
        save_steps=save_steps,
        ckpt_dir=ckpt_dir,
    )
    # 这里应该是读取了之前的best loss
    assert ckpt_mgr.get_best_loss() == 50.0

    assert not ckpt_mgr.need_save(step=1, loss=200)
    assert not ckpt_mgr.need_save_interval(step=1)
    assert not ckpt_mgr.need_save_best(loss=200)
    assert ckpt_mgr.need_save_best(loss=10)
    assert ckpt_mgr.need_save(step=100, loss=300)

    # 我们保存一个step同时也是best loss的，看看能不能正确的保存
    checkpoint = Checkpoint(
        config=generate_testing_config(),
        model={"dummy_model_key": "dummy_model_value"},
        optimizer={"dummy_optimizer_key": "dummy_optimizer_value"},
        scheduler={"dummy_scheduler_key": "dummy_scheduler_value"},
        grad_scaler={"dummy_grad_scaler_key": "dummy_grad_scaler_value"},
        step=100,
        dataloader=[{"dummy_dataloader_key": "dummy_dataloader_value"}],
        loss=10,
        num_tokens=0,
    )

    ckpt_mgr.save_checkpoint(checkpoint)
    assert ckpt_mgr.get_best_loss() == 10.0

    ckpt_files = list(ckpt_dir.iterdir())
    assert len(ckpt_files) == 3, "There should be three checkpoint files saved."
    assert "ckpt_step_0.tar" in [f.name for f in ckpt_files]
    assert "ckpt_step_100.tar" in [f.name for f in ckpt_files]
    assert "best_ckpt_step_100.tar" in [f.name for f in ckpt_files]

    ckpt_mgr.clear_checkpoints()
    ckpt_files = list(ckpt_dir.iterdir())
    assert len(ckpt_files) == 0, "All checkpoint files should be cleared."
    assert ckpt_mgr.get_best_loss() == float("inf"), (
        "Best loss should be reset to infinity after clearing checkpoints"
    )


# TODO: 还要再测试一个东西
# 就是目录里面已经有了一些checkpoint
# 能不能正确的初始化chekcpoint manager对象？
