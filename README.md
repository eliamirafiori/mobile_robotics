docker build -t mujoco-robotics .

docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -e PYOPENGL_PLATFORM=egl `
  -e NVIDIA_DRIVER_CAPABILITIES=all `
  mujoco-robotics python -c "import mujoco, gymnasium_robotics; print(mujoco.__version__)"

python swarm_coverage.py --compare          # adaptive vs. static, saves coverage_result.png
python swarm_coverage.py --gif swarm.gif    # animation
python swarm_coverage.py --drones 8 --grid 80 --comm-range 15 --no-events


