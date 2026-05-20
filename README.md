# Teman Isyarat — Project Manager

[![GitHub](https://img.shields.io/badge/project-temanisyarat-blue)](https://github.com/williamu04/temanisyarat)
[![Obsidian](https://img.shields.io/badge/editor-Obsidian-7C3AED)](https://obsidian.md)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

> **Teman Isyarat** (Indonesian for "Sign Friend") is an AI-powered Indonesian Sign Language (BISINDO) recognition system — developed by a student team at **Universitas Sebelas Maret (UNS)** under the Hibah Jarprak program, in partnership with **GERKATIN Solo**.

This repository is the **central project management and documentation hub** for the Teman Isyarat project. It uses an **Obsidian vault** structure to organize architecture decisions, technical specifications, task tracking, and team logs. The actual source code (landmark extraction, model training, mobile app, website, backend) lives in separate repositories.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Architecture Decisions](#architecture-decisions)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## About the Project

BISINDO (Bahasa Isyarat Indonesia) is the native sign language used by the Deaf community in Indonesia. Despite its widespread use, technological support for BISINDO recognition remains limited, creating communication barriers between Deaf and hearing individuals.

Teman Isyarat tackles this by building a **real-time gesture recognition system** capable of classifying **20 BISINDO vocabulary words** (Central Java dialect) using:

- **MediaPipe** for holistic landmark extraction (hands + pose)
- **GRU neural networks** for temporal gesture classification
- **Flutter + Kotlin** for an on-device Android application
- **Video augmentation** (FFmpeg) to expand a limited dataset

The project timeline spans **February — June 2026**, covering dataset collection with GERKATIN Solo, model development, mobile app implementation, user field testing, and community socialization.

### Vocabulary Set (20 Gestures)

| Category     | Words                          | **Total** |
| ------------ | ------------------------------ | --------- |
| **Pronouns** | Aku, Kamu, Dia                 | 3         |
| **Common**   | Salam, Terimakasih, Maaf, Nama | 4         |
| **Time**     | Hari ini, Besok                | 2         |
| **Color**    | Merah, Kuning                  | 2         |
| **Family**   | Ayah, Ibu                      | 2         |
| **Count**    | Satu, Dua, Tiga                | 3         |
| **Other**    | Teman, Buku                    | 2         |
| **Fruit**    | Apel, Pisang                   | 2         |


---

## Repository Structure

```
.
├── docs/
│   ├── adr/               # Architecture Decision Records (why)
│   └── spec/              # Technical Specifications (how)
├── tasks/
│   ├── backlog.md         # Administrative backlog & grant tasks
│   ├── logbook.md         # Academic logbook (course credit)
│   └── sprint-current.md  # Current sprint tracking
├── logs/                  # Individual team member progress logs
├── assets/                # Design mockups, screenshots, diagrams
├── .github/workflows/     # CI/CD (auto-merge, asset renaming)
├── CONTRIBUTING.md        # Contribution guidelines (INA)
└── README.md              # ← You are here
```

### Key Sections

| Path | Description |
|---|---|
| `docs/adr/` | 6 ADRs covering video strategy, GRU selection, gesture count, GERKATIN partnership, grant reporting, and extraction analysis |
| `docs/spec/` | 5 specs detailing MediaPipe extraction, mobile stack, FFmpeg augmentation, recording standards, and Flutter+MediaPipe integration |
| `tasks/` | Project backlog, academic logbook, and sprint planning |
| `logs/` | Per-member work logs (update with every PR) |
| `assets/` | UI mockups, documentation images, branding assets |

---

## Technology Stack

The broader Teman Isyarat software ecosystem spans multiple platforms:

### Machine Learning Pipeline

| Component | Technology |
|---|---|
| Landmark Extraction | MediaPipe Tasks Vision (Hand, Pose, Face Landmarker) |
| Video Processing | OpenCV-Python, FFmpeg |
| Model Architecture | Gated Recurrent Unit (GRU) — TensorFlow / Keras |
| Augmentation | FFmpeg (brightness ±0.1, horizontal flip, combinations — 6× per video) |

### Mobile Application

| Component | Technology |
|---|---|
| Cross-Platform UI | Flutter / Dart |
| Native Camera | Kotlin — CameraX |
| On-Device ML | MediaPipe Tasks via Android PlatformView |
| Communication | MethodChannel (Dart ↔ Kotlin) |

### Web Platform

| Component | Technology |
|---|---|
| Frontend | NextJS |
| Backend | Go (Golang) |

---

## Architecture Decisions

Key rationale documented in `docs/adr/`:

| ADR | Decision |
|---|---|
| [ADR-001](docs/adr/adr-001-pengambilan-video.md) | 5-day distributed video recording over single-day marathon to reduce participant fatigue |
| [ADR-002](docs/adr/adr-002-pilih-gru.md) | **GRU over LSTM/RNN** — better temporal sequence modeling for gesture recognition with fewer parameters |
| [ADR-003](docs/adr/adr-003-jumlah-gesture.md) | **20 gestures** — balanced between research significance and dataset feasibility |
| [ADR-004](docs/adr/adr-004-mitra-gerkatin.md) | **GERKATIN Solo** as community partner over PUSBISINDO for regional dialect alignment |
| [ADR-005](docs/adr/adr-005-laporan-kemajuan.md) | **70% stage-1 spending** to mitigate bureaucratic delays in grant fund disbursement |
| [ADR-006](docs/adr/adr-006-extraction-analysis.md) | **Drop face landmarks** — 84.5% NaN detection rate, reducing input from 252 → 153 dims |

---

## Getting Started

### Prerequisites

- [Obsidian](https://obsidian.md) (recommended for browsing/editing)

### Open the Vault

```bash
git clone git@github.com:williamu04/temanisyarat-manager.git
cd temanisyarat-manager
# Open this folder as an Obsidian vault
```

### Browse Documentation

Start with the architecture decisions to understand *why* key choices were made, then move to specs for *how* each component is built.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) (in Indonesian) for:

- Git workflow (fork → feature branch → PR)
- Commit message conventions (`feat`, `fix`, `docs`, etc.)
- Branch naming (`feature/`, `fix/`, `docs/`, etc.)
- PR template and review process
- **Mandatory:** update `logs/{nama}.md` with every pull request

---

## Team

| Name | Role |
|---|---|
| Dunwill William | Project Lead |
| Fredy Ramadhan | R&D |
| Hany Wachidatul Aisyah | R&D |
| Ian | R&D |
| Ivan | R&D |
| Kevin | R&D |
| Mutia | R&D |
| Saidah | R&D |

---

## Acknowledgments

- **GERKATIN Solo** — Dataset collection, validation, and field testing partnership
- **Universitas Sebelas Maret (UNS)** — Hibah Jarprak funding and academic support
- **Google MediaPipe Team** — Open-source landmark detection models

---

## License

This project is developed for academic purposes under the Hibah Jarprak program at Universitas Sebelas Maret.
