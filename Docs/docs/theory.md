# Theory

A brief overview of the physics models underlying WindGym simulations.

---

## Wake Effects

When wind flows through a turbine rotor, the extraction of kinetic energy creates a **wake**, a region of reduced wind speed and increased turbulence downstream. In a wind farm, these wakes propagate to downstream turbines, reducing their power output and increasing structural loads. Wake losses can reduce total farm power by 10–20% depending on layout and wind conditions.

**Wake steering** is a control strategy where upstream turbines are intentionally yawed (misaligned with the wind direction) to deflect their wakes away from downstream turbines. This is the core control action in WindGym: agents learn optimal yaw angles to maximize farm-level power production.

---

## Simulation Backends

WindGym supports two interchangeable simulation backends, selectable via the `backend` parameter. Both can be used with the same RL code, enabling a multi-fidelity workflow: train with the fast backend and validate with the high-fidelity one.

### DYNAMIKS: Dynamic Wake Meandering (High-Fidelity)

The default backend uses [DYNAMIKS](https://dynamiks.pages.windenergy.dtu.dk/dynamiks/), a multi-fidelity wind farm simulation framework implementing the Dynamic Wake Meandering (DWM) model. Key components include:

- **Ainslie deficit model**: Simulates the velocity deficit behind each turbine as a profile that evolves as it propagates downstream.
- **Hill Vortex wake steering**: Models the lateral deflection of wakes caused by yaw misalignment, based on the vortex curl induced by a yawed rotor.
- **Particle-based transport**: Wakes are represented as particle clouds that are advected and dispersed over time, capturing transient wake dynamics and meandering.

DYNAMIKS captures time-dependent wake behavior (wake recovery, meandering, dynamic interactions), making it suitable for studying temporal control strategies. For full details, see the [DYNAMIKS documentation](https://dynamiks.pages.windenergy.dtu.dk/dynamiks/).

### PyWake: Steady-State (Fast)

The alternative backend uses [PyWake](https://topfarm.pages.windenergy.dtu.dk/PyWake/), an analytical wake modeling framework. It employs:

- **Blondel-Cathelain 2020 Gaussian wake model** for velocity deficits.
- **Jimenez wake deflection model** for yaw-induced wake steering.
- **Crespo-Hernandez turbulence model** for added wake turbulence.

PyWake computes the flow field analytically at each step with no temporal evolution, making it significantly faster than DYNAMIKS but unable to capture transient wake dynamics.

### Comparison

| | DYNAMIKS (DWM) | PyWake |
|---|---|---|
| **Fidelity** | High (transient wake dynamics) | Engineering (steady-state) |
| **Speed** | Slower (particle simulation) | Fast (analytical) |
| **Wake meandering** | Yes | No |
| **Temporal dynamics** | Yes | No |
| **Best for** | Validation, dynamic control | Fast training, large parameter sweeps |

---

## Turbulence Modeling

WindGym supports Mann turbulence boxes, 3D spectral turbulence fields that provide realistic, spatially and temporally correlated wind fluctuations. Five turbulence modes are available:

- **MannLoad**: Load pre-generated Mann boxes from files.
- **MannGenerate**: Generate new Mann fields on-the-fly.
- **MannFixed**: Generate once and reuse (deterministic).
- **Random**: Fast random fluctuations (less realistic).
- **None**: Laminar flow (fastest, for testing).

---

## References

- DYNAMIKS: [Documentation](https://dynamiks.pages.windenergy.dtu.dk/dynamiks/) | [Repository](https://gitlab.windenergy.dtu.dk/DYNAMIKS/dynamiks)
- PyWake: [Documentation](https://topfarm.pages.windenergy.dtu.dk/PyWake/) | [Repository](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake)
