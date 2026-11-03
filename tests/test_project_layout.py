"""Offline checks for relocated resources and the self-contained worker bundle."""

from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace


def test_default_robot_resources_load_outside_repo(tmp_path, monkeypatch, lerobot_calibration_path):
    from motion_planner import SO101MotionPlanner
    from project_paths import MODEL_DIR, OFFSETS_PATH

    monkeypatch.chdir(tmp_path)
    planner = SO101MotionPlanner(lerobot_calibration_path=lerobot_calibration_path)
    assert planner.urdf == MODEL_DIR / "so101_brush_tip_nominal.urdf"
    assert planner.joint_offsets_path == OFFSETS_PATH
    assert len(planner.joint_names) == 5
    assert planner.viewer is None


def test_worker_build_context_is_complete_and_importable(monkeypatch):
    from remote import setup
    from remote.artifacts import code_version

    expected_version = code_version()
    run = subprocess.run
    commands = []

    def fake_docker(command, **kwargs):
        commands.append(command)
        if command[1] != "buildx":
            return
        context = Path(kwargs["cwd"])
        dockerfile = context / command[command.index("-f") + 1]
        # Check every COPY input, including globbed package files, against the
        # actual temporary context passed to Docker.
        for line in dockerfile.read_text().replace("\\\n", " ").splitlines():
            if line.startswith("COPY "):
                for source in shlex.split(line)[1:-1]:
                    assert list(context.glob(source)), source
        assert not list(context.rglob("runpod.json"))
        assert not (context / "src/caches").exists()
        assert not (context / "src/robot.py").exists()
        # Import only the copied sources, with this test environment's installed
        # dependencies; the working checkout must not fill gaps in the bundle.
        dependencies = [p for p in sys.path if p and (
            Path(p).name == "site-packages" or not Path(p).resolve().is_relative_to(setup.ROOT))]
        result = run([sys.executable, "-c", """
import sys
sys.path[:0] = sys.argv[1:]
import painting_optimization
from remote import worker
from remote.artifacts import code_version
from pathlib import Path
assert Path(worker.__file__).resolve().is_relative_to(Path.cwd().resolve())
assert Path(painting_optimization.__file__).resolve().is_relative_to(Path.cwd().resolve())
assert "painter" not in sys.modules
assert "robot" not in sys.modules
print(code_version())
""", str(context / "src"), *dependencies], cwd=context,
                     check=True, capture_output=True, text=True)
        assert result.stdout.strip().splitlines()[-1] == expected_version

    monkeypatch.setattr(setup.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 1024**3))
    monkeypatch.setattr(setup.subprocess, "run", fake_docker)
    image = setup.build_worker_image("example/frida-worker")
    assert image == f"example/frida-worker:{expected_version[:16]}"
    assert [command[1] for command in commands] == ["buildx", "run"]
