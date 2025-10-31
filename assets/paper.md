---
title: 'WindGym: A Reinforcement Learning Environment for Wind Farm Control'
tags:
  - Python
  - Reinforcement Learning
  - Wind energy
  - Wind Farm Control
authors:
  - name: Marcus Binder Nilsen
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0009-0001-5760-5225
    affiliation: "1" #
  - name: Julian Quick
    orcid: 0000-0002-1460-9808
    affiliation: "1" #
  - name: Teodor Olof Benedict Åstrand
    orcid: 	0009-0007-6400-2821
    affiliation: "1" # 
  - name: Ernestas Simutis
    affiliation: 1
  - name: Pierre-Elouan Mikael Réthoré
    orcid: 0000-0002-2300-5440
    affiliation: "1" #
affiliations:
 - name: Department of Wind and Energy Systems, Technical University of Denmark, Roskilde, Denmark
   index: 1
date: 23 September 2025
bibliography: paper.bib

---


<!-- https://joss.theoj.org/papers/10.21105/joss.06739
https://joss.theoj.org/papers/10.21105/joss.06746 -->

# Summary 

**WindGym** is an open-source Python package for reinforcement-learning (RL) based control of wind farms. It provides both single-agent and multi-agent environments, following the Gymnasium API for centralized controllers and the PettingZoo API for multi-agent settings, enabling drop-in use with mainstream RL frameworks [@gymnasium; @pettingzoo]. WindGym is built on top of DYNAMIKS, a multi-fidelity flow simulation framework, which allows users to seamlessly adjust between computational speed and physical fidelity within a single interface [@dynamiks].

WindGym lowers the barrier to reproducible research and benchmarking within the field of RL for wind farm control by standardizing interfaces and providing built-in examples, reward utilities, and tests. The package is MIT-licensed and comes with documentation, continuous integration, and ready-to-run training pipelines, making it straightforward for researchers to prototype, compare, and share RL-based wind farm control strategies.



# Statement of need

Wind energy is projected to play an increasingly important role in global energy production if the transition towards climate neutrality is to become true [@irena2022weto; @iea2021netzero]. Today, most wind turbines are placed closely together in wind farms to leverage shared infrastructure and reduce land use [@Vondelen2024]. However, this introduces what is known as the 'wake effect'. This occurs when an upstream turbine impedes the incoming flow, resulting in a decrease in wind speed and an increase in turbulence for the downstream turbines. This can result in decreased power output and increased structural loads [@Howland2020]. One way to mitigate this phenomenon is with wake steering, where turbines are intentionally misaligned with the wind to help steer the wake away from downstream turbines [@Annoni2018].

Developing control algorithms for wind farms is not a trivial task. One area that has been gaining increased interest is to use RL to help learn a control strategy based on a simulated wind farm environment [@abkar2023reinforcement; @goccmen2025data]. However, even though interest in this field is increasing, much of the work remains fragmented, with many researchers using custom simulators or failing to publish their code bases.

There already exist a lot of different options for simulating the behaviour of wind farms. These are typically divided into different levels of detail. For example, PyWake [@pywake] and Floris [@FLORIS] are able to simulate the steady-state flow over a full wind farm in the matter of milliseconds, but do not include the transient evolution and turbulent behavior of the wind. Alternatively, simulators such as FOXES [@foxes] and Floridyn [@floridyn] do include the transient evolution of the flow but do not account for the turbulent fluctuations of the wind. 

Our RL environment is built on Dynamiks [@dynamiks], which is a multi-fidelity framework. This means that it is possible to interchange the fidelity levels in the simulation with only minor changes to the code, and keeping the underlying flowsimulations in a unified framework. 

RL for wind farm control is a rapidly evolving field, but progress is hampered by a lack of standardized environments and reproducible benchmarks. WindGym addresses this gap by providing an RL-first framework that (i) follows the de facto RL APIs (Gymnasium for single-agent [@gymnasium] and PettingZoo for multi-agent [@pettingzoo]), (ii) abstracts different wind-farm simulation back-ends within a unified interface, and (iii) includes examples and tests to support reproducibility. By lowering the barrier to entry, WindGym enables systematic comparisons across algorithms, reward definitions, and simulator fidelity levels.

# Functionality

<!-- - **APIs & variants**  
  **Single-agent** follows the **Gymnasium** API (`reset/step`, `terminated/truncated`, `info`) for centralized controllers.  
  **Multi-agent** follows **PettingZoo** (Parallel API for simultaneous actions; AEC supported when sequential updates are desired), mapping **each turbine to an agent** with its own observation/action space [@gymnasium; @pettingzoo-aec; @pettingzoo-parallel]. This lets researchers prototype centralized vs. decentralized control with minimal code changes.

- **Physics back-ends**  
  Plug-compatible back-ends: **DYNAMIKS** for dynamic, higher-fidelity flow; **PyWake** for fast, analytical wakes and plant-level studies [@dynamiks; @pywake].

- **Reward utilities**  
  Built-ins for **power-based**, **baseline-normalized**, and **delta-power** rewards, with optional penalties (e.g., yaw travel/load proxies). Users can supply custom reward callables.

- **Training & evaluation**  
  Examples show end-to-end training (e.g., PPO) and evaluation/visualization; vectorized training is supported for throughput. WindGym integrates cleanly with standard RL codebases [@cleanrl; @ppo].

- **Reproducibility & QA**  
  Tests validate spaces, termination, and determinism toggles; CI and examples support consistent runs. -->

WindGym is designed to make it easy to prototype and benchmark reinforcement learning methods for wind farm control. It supports both centralized and decentralized formulations. In the single-agent variant, the environment follows the Gymnasium API, where a single controller issues actions for the entire farm. In the multi-agent variant, the environment follows the PettingZoo API, mapping each turbine to its own agent with separate observation and action spaces, allowing researchers to switch between control paradigms with only minor code changes [@gymnasium; @pettingzoo].

Behind the scenes, WindGym provides interchangeable physics back-ends. At present, it supports DYNAMIKS for dynamic, higher-fidelity transient simulations, and PyWake for fast, analytical wake models and farm-level studies [@dynamiks; @pywake]. These back-ends can be swapped without altering the RL setup, enabling researchers to trade off speed and fidelity as needed.

Reward specification is another central feature. WindGym includes utilities for common formulations such as raw power, baseline-normalized power, and delta-power rewards, as well as optional penalty terms. At the same time, users can easily plug in their own reward functions.

To lower the barrier to adoption, WindGym ships with examples that demonstrate end-to-end training and evaluation (e.g., PPO), along with visualization utilities for analyzing results. It integrates seamlessly with popular RL libraries [@cleanrl; @sb3], making it straightforward to slot into existing workflows.

Finally, reproducibility is a core concern. The environment is tested for consistency of observation and action spaces, correct termination behavior, and deterministic toggles. Continuous integration and curated examples help ensure that results can be reproduced across setups.

The full documentation of the library is available at [https://sys.pages.windenergy.dtu.dk/windgym/](https://sys.pages.windenergy.dtu.dk/windgym/)

<!-- # Acknowledgements

This work was supported by the Department of Wind and Energy Systems at the Technical University of Denmark. We also acknowledge the development of DYNAMIKS, which was instrumental in building WindGym. -->

# References
