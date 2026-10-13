# DEEPGUARD — DATASET PROVENANCE AND AUTHORIZATION REPORT
**Project:** DeepGuard — Explainable and Robust Spatial-Temporal Deepfake Video Detection
**Phase:** Phase 3 Dataset Integration
**Date:** Academic Year 2026
**Status:** **Formal Research Access Pending / Integration Infrastructure Ready**

---

## 1. PURPOSE AND ETHICAL MANDATE

In strict accordance with academic ethics, international data protection laws, and university guidelines:
* DeepGuard **does not** scrape unreleased, non-consensual, or private media.
* DeepGuard **does not** commit raw video files, facial biometric archives, or restricted research datasets to public GitHub repositories.
* DeepGuard **requires** that all evaluation benchmarks operate on peer-reviewed, authorized research datasets with consenting actors.

This document records the provenance, licensing, access procedures, and current authorization status for candidate deepfake video datasets.

---

## 2. CANDIDATE DATASET COMPARISON & ASSESSMENT

### 2.1 FaceForensics++ (FF++) — Primary Benchmark Candidate

* **Official Source:** Technical University of Munich (TUM) & Google Research
  *Paper:* *FaceForensics++: Learning to Detect Manipulated Facial Images* (Rössler et al., ICCV 2019)
  *Portal:* https://github.com/ondyari/FaceForensics
* **Access Procedure:**
  1. Complete and sign the official academic Google Form terms of use.
  2. Await manual review from the TUM research team.
  3. Receive an authorized Python download script with a temporary access token.
* **Academic Eligibility & License:**
  * Strict non-commercial, academic research license.
  * **Redistribution strictly prohibited** — raw videos must remain on secured institutional or personal cloud drives.
* **Dataset Composition:**
  * 1,000 pristine original YouTube videos (actors with facial consent).
  * 4 automated manipulation methods: **Deepfakes**, **Face2Face**, **FaceSwap**, **NeuralTextures** (plus FaceShifter in newer releases).
  * Three compression tiers: Raw (uncompressed), **c23 (light compression, default academic benchmark)**, and c40 (heavy compression).
* **Metadata & Pairing Availability:**
  * Videos indexed by numeric ID (`000` to `999`).
  * Direct original-to-manipulated mapping pairs are fully documented.
  * Pre-defined official benchmark train/val/test splits (720 train, 140 val, 140 test) available.
* **Storage Footprint:**
  * Full dataset: > 500 GB (raw).
  * c23 compressed partition (all methods): ~40–60 GB.
  * Curated student benchmark subset (e.g., 200 real + 200 fake c23 videos): ~3–5 GB (fits within student Google Drive quota).
* **Current Status for DeepGuard:**
  * **Application submitted; official credential approval is pending review.**

---

### 2.2 Celeb-DF (v2) — Secondary Generalization Candidate

* **Official Source:** SUNY Albany & Purdue University
  *Paper:* *Celeb-DF: A Large-scale Challenging Dataset for Deepfake Forensics* (Li et al., CVPR 2020)
  *Portal:* https://github.com/yuezunli/celeb-deepfakeforensic
* **Access Procedure:**
  1. Submit formal research request form with verified university email (`.edu` / academic institution).
  2. Agreement specifies research use only under the supervision of faculty.
* **Academic Eligibility & License:**
  * Non-commercial research only. Redistribution is prohibited.
* **Dataset Composition:**
  * 590 real celebrity YouTube videos.
  * 5,639 high-quality deepfake video sequences.
  * 300 additional testing YouTube videos.
* **Metadata & Pairing Availability:**
  * Actor identities explicitly documented via celebrity subject IDs (`id0` to `id58`).
  * Synthesis pairing file `Celeb-synthesis.txt` maps every fake video to its donor and recipient originals (e.g. `id0_id1_0001.mp4` derived from `id0/0001.mp4`).
* **Storage Footprint:**
  * Full dataset: ~16 GB compressed.
* **Current Status for DeepGuard:**
  * A 30-video mini development subset was used in the preliminary development baseline.
  * Full dataset integration script and subject-identity disjoint grouping are fully implemented in DeepGuard.

---

### 2.3 Deepfake Detection Challenge (DFDC Preview) — Fallback Candidate

* **Official Source:** AWS, Facebook/Meta, Partnership on AI, Microsoft
  *Portal:* Kaggle / AWS Open Data
* **Access Procedure:**
  * Kaggle competition account acceptance of competition rules.
* **Storage Footprint:**
  * Full DFDC: > 470 GB.
  * DFDC Preview: ~4.1 GB (4,119 video clips).
* **Metadata:**
  * Actor ID metadata provided in `metadata.json`.
* **Current Status for DeepGuard:**
  * Readily available as an operational fallback if TUM credentials experience prolonged administrative delay.

---

## 3. SELECTION DECISION & STATUS SUMMARY

| Candidate Dataset | Authorization Status | Primary Role in DeepGuard | Storage Footprint | Grouping Metadata |
| :--- | :--- | :--- | :--- | :--- |
| **FaceForensics++ (c23)** | **Application Pending** | Primary benchmark reproduction (matches reference paper) | ~3–5 GB (curated 400-video subset) | Original Video ID (`000`–`999`) |
| **Celeb-DF (v2)** | Available for Academic Use | Cross-dataset generalization benchmark | ~16 GB full / ~2 GB subset | Subject IDs (`id0`–`id58`) & Synthesis Pairs |
| **DFDC (Preview)** | Available via Kaggle | Fallback evaluation dataset | ~4.1 GB | Actor IDs in `metadata.json` |

### Explicit Research Blocker Statement
> [!IMPORTANT]
> **Active Blocker:** Full FaceForensics++ download credentials have not yet arrived from TUM.
> DeepGuard's Phase 3 pipeline is **architecturally and functionally ready** to ingest, validate, and train on the official FF++ c23 split the moment credentials are confirmed. Until then, DeepGuard operates on validated synthetic test fixtures and authorized subsets, preserving academic integrity without fabricating data.
