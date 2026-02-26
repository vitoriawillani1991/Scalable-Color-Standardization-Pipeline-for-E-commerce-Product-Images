# 🎨 Scalable Color Standardization Pipeline for E-commerce Product Images

## Executive Summary

This project transforms a subjective visual problem — inconsistent product image colors — into a **quantifiable, automated, and scalable computer vision system**.

By modeling color distributions in CIELAB space and aligning them statistically to a reference standard, this pipeline:

- Standardizes product imagery at scale  
- Reduces manual QA workload  
- Introduces objective quality metrics  
- Enables data-driven visual governance  

This is not just image correction — it is **operational automation powered by statistical modeling**.

---

# 🧩 Business Context

In large e-commerce catalogs:

- Images are captured under different lighting conditions  
- Vendors deliver inconsistent color tones  
- Manual Photoshop correction does not scale  
- No objective metric defines “visually correct”  

## Business Risks

- Brand inconsistency across listings  
- Increased QA overhead  
- Customer dissatisfaction due to perceived color mismatch  
- Potential negative impact on conversion rates  

The real challenge was not how to edit images.

It was how to make color consistency **measurable, scalable, and auditable**.

---

# 🧠 Data Science Framing

This problem was approached as:

> A distribution alignment task under perceptual color constraints.

Each product image is treated as a statistical distribution in CIELAB space.

We align:

- Mean (central tendency)  
- Standard deviation (contrast / dispersion)  

Between:

- A reference image (brand visual standard)  
- A target image (to be corrected)  

This reframes color correction as:

- Statistical normalization  
- Controlled transformation  
- Threshold-based classification  

---

# ⚙️ Methodology

## 1️⃣ Perceptual Color Space Transformation

Images are converted from RGB → CIELAB.

Why?

Because Lab separates:

- **L** → luminance  
- **a** → green ↔ red axis  
- **b** → blue ↔ yellow axis  

This allows independent control of brightness and chromaticity, making perceptual alignment more reliable than RGB-based correction.

---

## 2️⃣ Statistical Distribution Matching

For each channel:

```python
target = (channel - mean_current) * (std_ref / std_current) + mean_ref
```

This aligns both:

- Location (mean)  
- Spread (variance)  

The correction is blended using configurable strength parameters to prevent overcorrection artifacts:

- `LUMINANCE_STRENGTH`
- `COLOR_STRENGTH`

This creates a controlled statistical transfer instead of rigid normalization.

---

## 3️⃣ Two-Stage Correction Strategy

### Stage 1 — Distribution Alignment  
Aligns mean and variance of L, a, b channels.

### Stage 2 — Controlled Luminance Enforcement (Optional)  
Forces final L mean closer to reference while:

- Applying tolerance thresholds  
- Capping maximum allowed luminance shifts  
- Preventing highlight blowout  

This balances:

- Statistical consistency  
- Visual realism  
- Robustness to extreme inputs  

---

## 4️⃣ Automatic Quality Scoring

Instead of relying solely on visual inspection, the pipeline computes:

- ΔL (luminance deviation)  
- Δa, Δb (chromatic deviation)  
- ΔL_std (contrast deviation)  

Each image is automatically classified as:

- `OK`  
- `ACCEPTABLE`  
- `REVIEW`  

This converts subjective QA into threshold-based decision logic.

---

# 📊 Output Metrics

For every processed image, the system logs:

- Mean L/a/b before correction  
- Mean L/a/b after correction  
- Standard deviations  
- Absolute deviation from reference  
- Final quality classification  

All metrics are exported to a structured CSV file.

This enables:

- Performance tracking  
- Threshold tuning  
- Auditability  
- Data-driven governance  

Color consistency becomes a dataset — not an opinion.

---

# 📈 Business Impact

## 🔹 1. Manual QA Reduction

Instead of reviewing 100% of images:

Only statistically suspicious images are escalated.

This enables selective QA instead of full manual inspection.

---

## 🔹 2. Scalable Image Standardization

Hundreds or thousands of SKUs can be corrected automatically.

This makes brand-level visual consistency operationally sustainable.

---

## 🔹 3. Objective Visual Governance

The system introduces measurable thresholds for:

- Brightness deviation  
- Chromatic deviation  
- Contrast deviation  

This enables:

- Controlled experimentation  
- Continuous improvement  
- Policy-based image acceptance  

---

## 🔹 4. Marketplace Optimization

Consistent product imagery:

- Improves perceived professionalism  
- Reduces visual inconsistency across listings  
- Strengthens brand trust  

Visual consistency directly supports performance in digital retail environments.

---

# 🛠 Engineering Design Decisions

- Alpha masking isolates product pixels from background  
- Lab-based processing for perceptual alignment  
- Parameterized blending to avoid artifacts  
- Hard caps on luminance shifts for stability  
- Batch processing architecture  
- Automatic CSV logging  
- Automated routing of flagged images to REVIEW folder  

The system balances:

- Statistical rigor  
- Visual realism  
- Operational usability  

---

# 🧰 Tech Stack

- Python  
- OpenCV  
- NumPy  
- Statistical modeling in CIELAB  
- Batch file processing  
- Structured CSV logging  
