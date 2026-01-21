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

The goal of WindGym is to lower the barrier to reproducible research and benchmarking within the field of RL for wind farm control by standardizing interfaces and providing built-in examples, reward utilities, and tests. The package is MIT-licensed and comes with documentation, continuous integration, and ready-to-run training pipelines, making it straightforward for researchers to prototype, compare, and share RL-based wind farm control strategies.



# Statement of need


Wind energy is projected to play an increasingly important role in global energy production if the transition towards climate neutrality is to be realized [@irena2022weto; @iea2021netzero]. Today, most wind turbines are placed closely together in wind farms to leverage shared infrastructure and reduce land use [@Vondelen2024]. However, this introduces the wake effect, where an upstream turbine impedes the incoming flow, resulting in decreased wind speed and increased turbulence for downstream turbines. This can lead to decreased power output and increased structural loads [@Howland2020]. One way to mitigate this phenomenon is wake steering, where turbines are intentionally misaligned with the wind to help steer the wake away from downstream turbines [@Annoni2018].


Developing control algorithms for wind farms is not a trivial task. One area that has been gaining increased interest is using RL to learn control strategies based on simulated wind farm environments [@abkar2023reinforcement; @goccmen2025data]. However, even though interest in this field is increasing, much of the work remains fragmented, with many researchers using custom simulators or failing to publish their code bases. WindGym addresses this gap by providing an RL-first framework that follows the de facto RL APIs, abstracts different wind-farm simulation back-ends within a unified interface, and includes examples and tests to support reproducibility. By lowering the barrier to entry, WindGym enables systematic comparisons across algorithms, reward definitions, and simulator fidelity levels.


# State of the Field

Several options exist for simulating wind farm behaviour at different fidelity levels. PyWake [@pywake] and FLORIS [@FLORIS] simulate steady-state flow over a full wind farm in milliseconds but do not include transient evolution or turbulent behaviour. FOXES [@foxes] and Floridyn [@floridyn] include transient flow evolution but do not account for turbulent fluctuations. 

When we began developing WindGym, no existing package combined dynamic wind farm simulation with standard RL interfaces. While WFCRL [@WFCRL] has since emerged, providing RL environments built on Fastfarm [@fastfarm] and Floris, WindGym offers a distinct advantage: it is built on DYNAMIKS [@dynamiks], a multi-fidelity framework that allows users to interchange fidelity levels within a single codebase. This means researchers can train agents using fast, low-fidelity simulations and validate them with higher-fidelity models without changing their RL setup. Additionally, WindGym provides both single-agent and multi-agent environments through Gymnasium and PettingZoo APIs, whereas WFCRL currently focuses on multi-agent scenarios.


# Software Design

WindGym's architecture prioritizes simplicity and modularity. The core design centres on a single main environment file (`WindFarmEnv`) that encapsulates all essential logic for state management, action processing, and reward computation. The multi-agent variant (`MultiAgentWindFarmEnv`) is implemented as a thin wrapper around this core, mapping the centralized interface to per-turbine observations and actions. This approach minimizes code duplication and ensures consistent behaviour across control paradigms.

We deliberately adopted the Gymnasium and PettingZoo APIs as they represent the de facto standards in RL research. This decision lowers the barrier to entry for researchers already familiar with these interfaces and enables seamless integration with popular training libraries such as Stable-Baselines3 [@sb3] and CleanRL [@cleanrl].

The simulation back-end is abstracted behind a clean interface, allowing users to swap between DYNAMIKS for dynamic simulations and PyWake for steady-state analysis without modifying their RL code. This modularity supports diverse research directions, whether investigating large-scale RL training, robust control under uncertainty, or algorithm comparisons across fidelity levels.

Flexibility is maintained throughout: reward functions, observation spaces, and termination conditions are all configurable, enabling researchers to adapt the environment to their specific research questions rather than being constrained by rigid defaults.



# Functionality


WindGym supports both centralized and decentralized control formulations. In the single-agent variant, a single controller issues actions for the entire farm following the Gymnasium API. In the multi-agent variant following the PettingZoo API, each turbine maps to its own agent with separate observation and action spaces, allowing researchers to switch between paradigms with minimal code changes.

The package provides interchangeable physics back-ends: DYNAMIKS for dynamic, higher-fidelity transient simulations, and PyWake for fast, analytical wake models. These can be swapped without altering the RL setup, enabling researchers to trade off speed and fidelity as needed.

Reward specification is a central feature. WindGym includes utilities for common formulations such as raw power, baseline-normalized power, and delta-power rewards, as well as optional penalty terms. Users can also implement custom reward functions.

Finally, reproducibility is a core concern. The environment is tested for consistency of observation and action spaces, correct termination behavior, and deterministic toggles. Continuous integration and curated examples help ensure that results can be reproduced across setups.

The full documentation of the library is available at [https://sys.pages.windenergy.dtu.dk/windgym/](https://sys.pages.windenergy.dtu.dk/windgym/)


# Research Impact Statement

WindGym is still relatively new, but has gained traction within the wind energy research community, and as of January 2026, the repository has accumulated 48 stars on GitHub. To our knowledge, four research papers are currently in submission that utilize WindGym as their experimental platform, demonstrating its adoption for novel research contributions in RL-based wind farm control.

The package is designed for community readiness: comprehensive documentation explains core concepts and usage patterns, worked examples demonstrate training and evaluation workflows, and an extensive test suite ensures reliability across updates. We actively encourage external contributions through our Github/GitLab repository. 

# AI Usage Disclosure

The WindGym codebase was initiated before the widespread adoption of large language models and coding assistants, with the foundational architecture developed without AI assistance. As these tools matured, they were incorporated into the development workflow in the following ways: refactoring existing code for improved consistency and maintainability, generating documentation content, and developing a substantial portion of the unit test suite. All AI-generated code was reviewed and validated by human developers before integration.

For this paper, AI tools were used to provide feedback on clarity and wording during the drafting process. Grammarly was used for grammar and style checking. No content was generated wholesale by AI without human review and revision.


# References
