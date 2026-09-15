#!/usr/bin/env bash
#SBATCH --job-name=alpha20-render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=render-%j.out
#SBATCH --error=render-%j.err
#SBATCH --mail-type=NONE

set -euo pipefail
export MPLCONFIGDIR="${PWD}/.mplconfig"
python3 scripts/render_alpha20_dual.py
ffmpeg -y -framerate 20 -i frames_4k/frame_%04d.png \
  -c:v libx264 -preset slow -crf 10 -pix_fmt yuv420p -movflags +faststart \
  alpha20_m3_sst_urans_schlieren_mach_4k.mp4
