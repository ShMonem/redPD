This repository hosts projective dynamics implementations that are used to serve comparisions for different reduction subspaces for the physics based simulations.

- We used the code published 2018 for [Hyper-reduced projective dynamics](https://replicability.graphics/papers/10.1145-3197517.3201387/index.html), under MIT license, as a base code.
- In the old original code by C. Brandt, linear-blend reduction subspaces can be computed.
- Current code has deviated a lot from the original.
  - Provides snapshots/frames to compute snapshots-based bases.
  - Reads and applies position reduction using snapshots-bases if provided in .bin format.
  - Stores resutls in both .png and .off format for visual and numerical comparisons. , visit [animSnapBases](https://github.com/ShMonem/animSnapBases).
  - For more information on computing snapshots-bases for real-time interactive simulations

Repository:
- https://github.com/ShMonem/redPD

Developers:
- [Shaimaa Monem](https://orcid.org/0009-0008-4038-3452) (2021--)
- [Max Planck Institute for Dynamics of Complex Technical Systems](https://www.mpi-magdeburg.mpg.de/2316/en), Magdeburg, Germany.

Scientific Advisors:
- [Peter Benner](https://orcid.org/0000-0003-3362-4103) and [Christian Lessig](https://orcid.org/0000-0002-2740-6815) (2021-2024).

Reproducibility:
- This repository can be used to reproduce results in *Improved-Projective-Dynamics-Global-Using-Snapshots-based-Reduced-Bases* SIGGRAPH23 [1st place student competition award-winning paper](https://dl.acm.org/doi/10.1145/3588028.3603665).
- Refer to `README.test.md` for details.