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
  python src/run_swarm.py --mode headless --scenario charging --algo charge_test --steps 1500
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

## FIXES

- [x] Non fare la media MEAN tra le mappe di incertezza, ma dare priorità alla certezza MAX dei droni
- [] FINIRE AREA COVERAGE
- [] IMPLEMENTARE STRATEGIA ERGODIC
- [] IMPLEMENTARE RL

## INFO

- GPS per zone aperte, GPA non accurato per zone interne
- Utilizzo di landmarks o oggetti di riferimento (es. staccionata)
- Waypoints per la navigazione del drone
- A livello industriale si approaccia a task di computer vision
- MAPPE RANDOMIZZATE (DIMENSIONE E TOPOLOGIA, SEMPRE PLANARE)
- NUMERO DI STEP NECESSARI PER IL COMPLETAMENTO DEL TASK
- MASSIMO PERCORSO DAI DRONI
- COLLISION AVOIDANCE
- BATTERY MANAGEMENTE (NUMERO DI VOLTE NECESSARIO PER LA CARICA)

## PER PUBBLICAZIONE ROBOTICS AND AUTOMATION LETTERS

- [] RICOSTRUZIONE 3D
- [] METRICHE QUALITATIVE TRA POINT CLOUD A MESH
