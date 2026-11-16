# Elia Mirafiori - Mobile Robotics

## Build Image

```powershell
docker build -t drone-swarm-mujoco .
```

## Headless Mode

### Heuristic

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python swarm_coverage.py --mode headless --algo voronoi --steps 1500
```

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python src/run_swarm.py --mode headless --algo voronoi --steps 1500
```

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python src/run_swarm.py --mode headless --scenario charging --algo battery_patrol --steps 1500
```

### RL

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python rl_swarm_env.py --train --timesteps 150000 --scenario flat
```

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=egl `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python rl_swarm_env.py --mode headless --scenario flat
```

## Visual Mode

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=glfw `
  -e DISPLAY=:0 `
  -v "\\wsl.localhost\Ubuntu\mnt\wslg\.X11-unix:/tmp/.X11-unix" `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python swarm_coverage.py --mode visual --algo frontier
```

```powershell
docker run --rm -it --gpus all `
  -e MUJOCO_GL=glfw `
  -e DISPLAY=:0 `
  -v "\\wsl.localhost\Ubuntu\mnt\wslg\.X11-unix:/tmp/.X11-unix" `
  -v "${PWD}:/workspace" `
  drone-swarm-mujoco `
  python rl_swarm_env.py --mode visual --scenario flat
```
