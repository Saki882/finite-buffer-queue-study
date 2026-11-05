# Finite-Buffer Queue Performance and Packet-Loss Clustering under Correlated Traffic Arrivals and Service Times

## Overview

This repository contains the source code, experimental data, statistical analysis, and figures for the study **“Finite-Buffer Queue Performance and Packet-Loss Clustering under Correlated Traffic Arrivals and Service Times.”**

The study investigates how correlated packet arrivals and correlated service times affect the performance of a finite-buffer, single-server FIFO queue. Particular attention is given to packet loss, queue occupancy, packet delay, loss-run clustering, and recovery behavior.

A custom C++ discrete-event simulator was developed to evaluate the queue under different traffic conditions and service-time correlation levels.

## Repository Structure

```text
finite-buffer-queue-study/
│
├── README.md
│
├── source_code/
│   ├── ref_sim.cpp
│   ├── run_ref_grid.py
│   ├── analyze_ref.py
│   ├── additional_tests.py
│   └── repair_overflow.py
│
├── data/
│   ├── all.csv
│   └── run_manifest.json
│
├── analysis/
│   ├── summary.csv
│   ├── effect_of_phi.csv
│   ├── welch_phi_comparisons.csv
│   ├── welch_additional_comparisons.csv
│   └── anova_reports/
│
└── figures/
    └── PNG figures
```

## Queue Model and Experimental Parameters

The simulated system is a single-server FIFO queue with finite capacity and a tail-drop policy.

| Parameter | Configuration |
|---|---|
| Queue discipline | FIFO |
| Buffer capacity (K) | 10, 20, 50 packets |
| Server capacity | 10 Mbit/s |
| Packet sizes | 100 and 1,100 bytes |
| Packet-size probabilities | 0.5 and 0.5 |
| Offered load (ρ) | 0.5, 0.7, 0.9 |
| Service-time correlation (φ) | -0.6, 0, 0.3, 0.6, 0.9 |
| Arrival processes | Poisson and two-state MMPP |
| MMPP burstiness factors | h = 2, 3, 5 |
| Replications | 20 per configuration |
| Simulation duration | 5,000 seconds |
| Warm-up period | 100 seconds |

The buffer capacity K includes the packet currently in service.

### Service-Time Correlation

Packet sizes follow a two-state Markov chain with equal stationary probabilities.

The correlation parameter φ controls the probability of switching between packet-size states:

$$
P_{\mathrm{switch}}=\frac{1-\phi}{2}
$$

Positive values of φ increase the tendency for consecutive packets to have the same size, while negative values increase the tendency to alternate between packet sizes.

The packet-size state advances when service begins and remains unchanged while the server is idle.

### Arrival Processes

Two types of arrival processes are considered:

- **Poisson arrivals:** Used as the non-bursty reference case.
- **Two-state MMPP arrivals:** Used to investigate the effects of bursty traffic.

For the MMPP cases, the high-state offered load is calculated as:

$$
\rho_H=\frac{2\rho h}{1+h}
$$

where ρ is the average offered load and h is the burstiness factor.

Some high-state offered-load configurations exceed 1 and are included as burst-stress conditions.

## Experimental Design

The experiment includes:

- 4 arrival-process configurations: Poisson and MMPP with h = 2, 3, and 5.
- 3 offered-load levels.
- 3 buffer capacities.
- 5 service-time correlation levels.

This produces:

$$
4 \times 3 \times 3 \times 5 = 180
$$

experimental configurations.

Each configuration uses 20 independent replications, giving a total of:

$$
180 \times 20 = 3,600
$$

simulation runs.

## Performance Metrics

The analysis considers the following performance measures:

- Packet loss ratio
- Mean number of packets in the system
- Mean packet delay
- 95th-percentile packet delay
- Packet-loss burst ratio
- Mean packet-loss run length
- Percentage of lost packets occurring in runs of at least two consecutive losses
- Queue recovery time

Metrics based on loss runs exclude replications with zero packet losses. Recovery-time calculations use completed recovery episodes.

## Software Requirements

The experiments were conducted using:

- Ubuntu 24.04.2 LTS
- GCC/G++ 13.3.0
- C++17
- Python 3

Python analysis scripts may require additional scientific-computing packages. Install the required packages in a Python virtual environment if necessary.

## Running the Simulator

### 1. Compile the C++ simulator

From the repository root:

```bash
g++ -O2 -std=c++17 -Wall -Wextra -pedantic source_code/ref_sim.cpp -o source_code/ref_sim
```

### 2. Run the experimental grid

```bash
python3 source_code/run_ref_grid.py --bin ./source_code/ref_sim --out ./data/new_experiment --workers 2
```

This runs the experimental configurations using two workers and writes the new experiment output to `data/new_experiment`.

### 3. Run the analysis

To analyze the archived dataset:

```bash
python3 source_code/analyze_ref.py --csv data/all.csv
```

The analysis scripts and archived statistical outputs are provided in this repository.

## Validation

The simulator was validated against the analytical results of an M/M/1/K queue.

For the baseline configuration with:

- Offered load: ρ = 0.9
- Buffer capacity: K = 20

the simulated packet-loss probability was:

$$
0.013622 \pm 0.000247
$$

The corresponding analytical result was:

$$
0.013651
$$

The simulated mean number of packets in the system was:

$$
6.4297 \pm 0.0314
$$

The analytical result was:

$$
6.4199
$$

Both analytical values fall within the corresponding simulation 95% confidence intervals.

## Dataset and Reproducibility Notes

The `data/` directory contains the archived experimental dataset and its run manifest.

The final dataset contains 3,600 runs. During dataset correction, 116 runs were repeated after restoring missing right-censored full-buffer duration values. The remaining 3,484 runs were retained from the earlier experiment.

Therefore, the final dataset combines retained earlier runs with the 116 repeated runs. It should not be interpreted as 3,600 runs generated using one unchanged simulator binary.

The run manifest provides information about the archived experiment.

### Source and Binary Checksums

SHA-256 checksum of the archived C++ source file:

```text
50eb17f7dafaecdb63af6705465ebdace08f5e2b1b7212646fdf604737cc9b08
```

SHA-256 checksum of the audited simulator binary:

```text
9f7d269f316fa7203ff11643001c327b9e80addf353742e07887de109f98d4ff
```

These checksums can be used to verify the corresponding files, provided the archived files are unchanged.

## Statistical Analysis

The repository includes statistical comparisons of service-time correlation levels, including:

- Welch's independent-samples t-tests
- 95% confidence intervals
- Hedges' g effect sizes
- Holm correction for multiple comparisons
- Additional packet-loss clustering and recovery-time comparisons
- Exploratory ANOVA reports

The statistical output files are provided in the `analysis/` directory.

## Figures

The `figures/` directory contains the PNG figures generated from the experimental results. These figures illustrate the effects of traffic burstiness, buffer capacity, offered load, and service-time correlation on queue performance.

## Limitations

The results are based on a single-server finite-buffer queue model. They should therefore be interpreted within the assumptions of the simulated system.

The study does not directly model a complete real-world communication network. Further work could investigate more complex network topologies, scheduling policies, and traffic models.



## Contact

**Saqib Hussain**  
Email: saqibhussainit882@gmail.com  
GitHub: https://github.com/Saki882
