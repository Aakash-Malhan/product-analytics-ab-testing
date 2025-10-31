# Product Analytics: Cohorts, Funnel & A/B Testing 

**Live demo:** https://huggingface.co/spaces/aakash-malhan/product-analytics  
**Dataset:** MovieLens 1M (as a proxy for consumer engagement)

A full, FAANG-style product analytics app:
- Event processing → **cohorts** (weekly retention)
- **Activation funnel** (first 7 days)
- **A/B testing** with **CUPED** variance reduction + **SRM** check
- Gradio UI, Plotly charts, reproducible pipeline

<img width="1409" height="801" alt="Screenshot 2025-10-30 200837" src="https://github.com/user-attachments/assets/f65496bd-a651-4ea4-bf64-b6e2c82c4d3c" />
<img width="1416" height="382" alt="Screenshot 2025-10-30 200910" src="https://github.com/user-attachments/assets/cddca6bd-d216-473f-8e24-ec5ecab956c7" />
<img width="1502" height="730" alt="Screenshot 2025-10-30 200944" src="https://github.com/user-attachments/assets/d3c7ac5f-2ba6-4bbf-97b4-3bf5cceaa55f" />



## Why this matters (Meta/FAANG style)
- Shows ability to **define product metrics**, run **experiments**, and **tell data stories**
- Demonstrates **variance reduction (CUPED)** and **experiment health (SRM)**
- End-to-end: dataset → pipeline → stats → interactive app

---

## Key Results (from demo run)
- **Activation:** 6,040 → 6,020 users hit activation (≈ **99.7%**)  
- **Day-7 retention:** 6,020 → **174** users (≈ **2.9%**)
- **A/B Experiment:** ~**+11.5%** lift in 7-day views (p < 0.001), **SRM p ≈ 0.17** (healthy)
- **Distribution:** heavy right-tail (power users), treatment shifts curve right

> Interpretation: onboarding is smooth but **habit formation is weak**; the treatment **significantly lifts engagement**, validated with **CUPED**.
