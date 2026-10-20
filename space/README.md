---
title: quantity-lab
emoji: 📏
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: "How small floor-plan errors distort area and room counts"
---

# quantity-lab showcase

How small geometry errors turn into wrong takeoff quantities. Pick a synthetic floor plan and a mode: the annotated rooms (oracle), a controlled error (rooms 2 or 5 px too small or too big, a missing divider, a spurious line), a classical OpenCV room baseline, or the pretrained CubiCasa5K segmentation model (its output on synthetic plans, computed ahead of time). The tables give measured results on ten held-out CubiCasa5K test plans, as numbers only. Everything was computed ahead of time; the page runs no model.

Code (MIT): [github.com/dzakybd/quantity-lab](https://github.com/dzakybd/quantity-lab). Write-up: [dzaky.ai.id](https://dzaky.ai.id/en/blog/quantity-lab-takeoff-errors).

CubiCasa5K: Kalervo et al., SCIA 2019, https://zenodo.org/records/2613548 (CC BY-NC-SA 4.0). No CubiCasa5K image is included in this Space.
