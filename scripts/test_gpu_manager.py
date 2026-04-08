#!/usr/bin/env python3
"""
Test script for gpu_manager.py

Verifies that gpu_manager can:
1. Load and initialize properly
2. Parse configs and create experiments
3. Read metrics and determine stages
4. Handle edge cases (None values, missing files, etc.)

Usage: python test_gpu_manager.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gpu_manager import (
    Experiment,
    ExperimentManager,
    get_last_value,
    get_free_gpus,
    should_terminate_early,
    load_baseline_curve,
    get_baseline_final_90,
)


def test_get_last_value():
    """Test the get_last_value utility function."""
    print("\n=== Testing get_last_value ===")

    # Test with list
    assert get_last_value({'key': [1.0, 2.0, 3.0]}, 'key') == 3.0
    print("  [OK] List value")

    # Test with missing key
    assert get_last_value({}, 'key') is None
    print("  [OK] Missing key returns None")

    # Test with non-list value
    assert get_last_value({'key': 42.0}, 'key') is None
    print("  [OK] Non-list value returns None")

    # Test with empty list
    assert get_last_value({'key': []}, 'key') is None
    print("  [OK] Empty list returns None")


def test_experiment_stage_determination():
    """Test Experiment.determine_stage() with various t_env values."""
    print("\n=== Testing Experiment.determine_stage ===")

    exp = Experiment(
        exp_id='test_001',
        tag='test_exp',
        config={
            'q_tot_stage_steps': 500000,
            'reward_stage_steps': 100000,
            't_max': 600000,
        },
        gpu_id=0,
    )

    # Mock get_t_env to return specific values
    def mock_t_env(value):
        def getter():
            return value
        exp.get_t_env = getter
        return value

    # Test each stage
    mock_t_env(0)
    assert exp.determine_stage() == 'stage1'
    print(f"  [OK] t_env=0 -> stage1")

    mock_t_env(499999)
    assert exp.determine_stage() == 'stage1'
    print(f"  [OK] t_env=499999 -> stage1")

    mock_t_env(500000)
    assert exp.determine_stage() == 'stage2'
    print(f"  [OK] t_env=500000 -> stage2")

    mock_t_env(599999)
    assert exp.determine_stage() == 'stage2'
    print(f"  [OK] t_env=599999 -> stage2")

    mock_t_env(600000)
    # t_max=600000, so t_env=600000 is not < t_max, returns 'done'
    assert exp.determine_stage() == 'done'
    print(f"  [OK] t_env=600000 -> done (at t_max)")

    mock_t_env(700000)
    assert exp.determine_stage() == 'done'
    print(f"  [OK] t_env=700000 -> done")

    # Test terminal states bypass stage calculation
    for terminal_state in ['done', 'killed', 'crashed']:
        exp.stage = terminal_state
        assert exp.determine_stage() == terminal_state
        print(f"  [OK] Already {terminal_state} returns '{terminal_state}' directly")


def test_experiment_t_env_none_handling():
    """Test that get_t_env handles None from get_last_value gracefully."""
    print("\n=== Testing t_env None handling ===")

    exp = Experiment(
        exp_id='test_002',
        tag='test_exp',
        config={},
        gpu_id=0,
    )

    # Mock get_latest_metrics to return None (no metrics yet)
    exp.get_latest_metrics = lambda: None
    t_env = exp.get_t_env()
    assert t_env == 0
    print(f"  [OK] get_t_env() with no metrics returns 0")

    # Now determine_stage should work without error
    exp.stage = 'pending'
    stage = exp.determine_stage()
    print(f"  [OK] determine_stage() with t_env=0 works, stage={stage}")


def test_experiment_get_battle_won():
    """Test Experiment.get_battle_won() with various metric states."""
    print("\n=== Testing get_battle_won ===")

    exp = Experiment(
        exp_id='test_003',
        tag='test_exp',
        config={},
        gpu_id=0,
    )

    # Mock get_latest_metrics
    exp.get_latest_metrics = lambda: {'test_battle_won_mean': [0.1, 0.5, 0.8]}
    assert exp.get_battle_won() == 0.8
    print("  [OK] Returns latest value from list")

    exp.get_latest_metrics = lambda: None
    assert exp.get_battle_won() == 0.0
    print("  [OK] Returns 0 when metrics is None")

    exp.get_latest_metrics = lambda: {'test_battle_won_mean': []}
    assert exp.get_battle_won() == 0.0
    print("  [OK] Returns 0 when list is empty")


def test_experiment_manager_init():
    """Test ExperimentManager initialization with minimal args."""
    print("\n=== Testing ExperimentManager initialization ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal test config
        config_file = Path(tmpdir) / 'test_queue.yaml'
        config_file.write_text("""
queue:
  - tag: test_exp_1
    batch_size: 32
  - tag: test_exp_2
    batch_size: 64
""")

        baseline_dir = Path(tmpdir) / 'baseline'
        baseline_dir.mkdir()

        # Create mock args
        args = MagicMock()
        args.exp_configs = [str(config_file)]
        args.baseline_dir = str(baseline_dir)
        args.check_interval = 60

        manager = ExperimentManager(args)

        assert len(manager.queue) == 2
        print(f"  [OK] Loaded {len(manager.queue)} experiments")

        # Baseline may be empty if no tensorboard events exist
        print(f"  [OK] Baseline final_90={manager.baseline_final_90:.3f} (may be 0 if no data)")

        assert len(manager.running) == 0
        print("  [OK] No running experiments initially")


def test_should_terminate_early():
    """Test the early termination logic."""
    print("\n=== Testing should_terminate_early ===")

    # Create baseline curve with realistic win rates (0.3 to 0.8 range)
    baseline_curve = {i: 0.3 + i * 0.000001 for i in range(500000)}

    exp = Experiment(
        exp_id='test_004',
        tag='test_early_stop',
        config={
            'q_tot_stage_steps': 100000,
            'reward_stage_steps': 50000,
        },
        gpu_id=0,
    )

    # Test stage 1/2 early termination - win rate below 50% of baseline
    exp.get_t_env = lambda: 50000
    exp.get_battle_won = lambda: 0.1  # Very low
    exp.last_time_above_baseline_90 = 0

    should_stop, reason = should_terminate_early(exp, baseline_curve)
    assert should_stop == True
    print(f"  [OK] Stage1/2 low win rate triggers early stop: {reason}")

    # Test stage 3 early termination - win rate below 90% for > 200K steps
    # stage2_end = 150000, so t_env=400000 gives stage3_steps=250000 > 200000
    exp.get_t_env = lambda: 400000
    exp.get_battle_won = lambda: 0.35  # Below 90% of baseline at t=400000

    should_stop, reason = should_terminate_early(exp, baseline_curve)
    assert should_stop == True
    print(f"  [OK] Stage3 low win rate triggers early stop: {reason}")

    # Test no termination when performance is good
    exp.get_t_env = lambda: 400000
    exp.get_battle_won = lambda: 0.7  # Good performance, above 90% threshold

    should_stop, reason = should_terminate_early(exp, baseline_curve)
    assert should_stop == False
    print(f"  [OK] Good performance does not trigger early stop")

    # Test empty baseline curve - should never terminate
    exp.get_t_env = lambda: 50000
    exp.get_battle_won = lambda: 0.0

    should_stop, reason = should_terminate_early(exp, {})
    assert should_stop == False
    print(f"  [OK] Empty baseline curve does not trigger early stop")


def test_get_free_gpus():
    """Test GPU availability checking."""
    print("\n=== Testing get_free_gpus ===")

    try:
        free_gpus = get_free_gpus(min_free_mb=100)  # Low threshold for testing
        print(f"  [OK] Found {len(free_gpus)} free GPU(s): {free_gpus}")
    except Exception as e:
        print(f"  [WARN] get_free_gpus failed (may be expected on this machine): {e}")


def test_load_baseline_curve_missing_dir():
    """Test baseline curve loading with missing directory."""
    print("\n=== Testing load_baseline_curve (missing dir) ===")

    curve = load_baseline_curve('/nonexistent/baseline/dir')
    assert curve == {}
    print("  [OK] Missing directory returns empty dict")


def test_full_stage_transition_sequence():
    """Test the complete stage transition sequence."""
    print("\n=== Testing full stage transition sequence ===")

    exp = Experiment(
        exp_id='test_005',
        tag='full_test',
        config={
            'q_tot_stage_steps': 100,
            'reward_stage_steps': 50,
            't_max': 150,
        },
        gpu_id=0,
    )

    # stage2_end = 100 + 50 = 150 = t_max
    # So: t_env < 100 -> stage1, t_env < 150 -> stage2, t_env >= 150 -> done
    t_env_values = [0, 50, 99, 100, 130, 149, 150, 200]
    expected_stages = ['stage1', 'stage1', 'stage1', 'stage2', 'stage2', 'stage2', 'done', 'done']

    for t_env, expected in zip(t_env_values, expected_stages):
        exp.get_t_env = lambda v=t_env: v
        exp.stage = 'pending'  # Reset to allow determine_stage to compute
        result = exp.determine_stage()
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] t_env={t_env:3d} -> {result:8s} (expected {expected})")
        assert result == expected, f"t_env={t_env}: got {result}, expected {expected}"


def test_get_baseline_final_90():
    """Test get_baseline_final_90 function."""
    print("\n=== Testing get_baseline_final_90 ===")

    # Empty curve
    assert get_baseline_final_90({}) == 0.0
    print("  [OK] Empty curve returns 0.0")

    # Normal curve
    curve = {0: 0.3, 100: 0.4, 200: 0.5, 300: 0.6, 400: 0.7}
    final_90 = get_baseline_final_90(curve)
    assert final_90 == 0.9 * 0.7  # 0.9 * max
    print(f"  [OK] Curve with max=0.7 gives final_90={final_90:.3f}")


def main():
    print("=" * 60)
    print("GPU Manager Test Suite")
    print("=" * 60)

    tests = [
        test_get_last_value,
        test_experiment_stage_determination,
        test_experiment_t_env_none_handling,
        test_experiment_get_battle_won,
        test_experiment_manager_init,
        test_should_terminate_early,
        test_get_free_gpus,
        test_load_baseline_curve_missing_dir,
        test_full_stage_transition_sequence,
        test_get_baseline_final_90,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
