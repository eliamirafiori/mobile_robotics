## Build Image

docker build -t drone-swarm-mujoco .

## Headless Mode

docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python swarm_coverage.py --mode headless --algo voronoi --steps 1500

## Visual Mode

docker run --rm -it --gpus all `
  -e MUJOCO_GL=glfw `
  -e DISPLAY=:0 `
  -v "\\wsl.localhost\Ubuntu\mnt\wslg\.X11-unix:/tmp/.X11-unix" `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python swarm_coverage.py --mode visual --algo frontier