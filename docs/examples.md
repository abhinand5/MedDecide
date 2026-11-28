# Osler — Real-World Clinical Decision Examples

These freshly written, synthetic clinical vignettes illustrate how **Osler-9B** and **Osler-4B** handle diverse single-pass medical decisions (`noul`, `choice`, `score`) without text generation — covering negation, noisy EHR distractors, metabolic differentials, drug interactions, numeric lab thresholds, abstention when evidence is missing, minimal-pair flips, and clinical trial eligibility.

All outputs below were measured directly from the standalone merged checkpoints via `AutoModel.from_pretrained("abhinand/Osler-9B", trust_remote_code=True)` and `AutoModel.from_pretrained("abhinand/Osler-4B", trust_remote_code=True)` (one question per call, built-in dev calibration).

---

## 1. Negation & Ruled-Out Diagnosis (Chest Pain / PE)

**State:**
> 54yo F presented to ED with substernal burning chest discomfort after dinner. Given recent 6-hour flight, CTA chest was obtained and showed NO pulmonary embolism and no aortic dissection. High-sensitivity troponin T was <6 ng/L at 0h and 3h; ECG normal sinus rhythm. Symptoms resolved completely after viscous lidocaine and IV pantoprazole. Discharged on daily omeprazole.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *Does this note confirm a diagnosis of pulmonary embolism?* | `noul` | `p(yes) = 0.043` | `p(yes) = 0.027` |
| *What is the most likely final diagnosis explaining this visit?* (`pe` / `nstemi` / `gerd` / `pneumothorax`) | `choice` | **`gerd`** (`0.993`) | **`gerd`** (`0.991`) |

---

## 2. Messy EHR Distractor ("Cardiac Asthma" vs. Decompensated HFrEF)

**State:**
> 71yo M with childhood asthma (no inhaler use in 15 yrs), HTN, and ischemic cardiomyopathy (EF 25%) admitted for 4 days of worsening orthopnea, 3-pillow sleep, and 4 kg weight gain. Exam: JVP 14 cm, bibasilar crackles, 2+ pitting edema to knees; mild expiratory wheeze ('cardiac asthma'). Two albuterol nebulizers in ED gave no relief. BNP 2,650 pg/mL. CXR: vascular congestion and Kerley B lines.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *What is the primary driver of this patient's acute dyspnea?* (`asthma` / `hfref_exacerbation` / `copd` / `pneumonia`) | `choice` | **`hfref_exacerbation`** (`0.997`) | **`hfref_exacerbation`** (`0.992`) |
| *Is IV loop diuretic therapy clinically indicated?* | `noul` | `p(yes) = 0.956` | `p(yes) = 0.953` |

---

## 3. Metabolic Differential (MALA vs. DKA)

**State:**
> 66yo F with T2DM on metformin 1000 mg BID and CKD3b had 3 days of gastroenteritis with poor PO intake. Labs: Na 138, K 5.4, Cl 98, HCO3 10, BUN 58, Cr 4.1 (baseline 1.5), glucose 114 mg/dL, beta-hydroxybutyrate 0.3 mmol/L (normal), venous pH 7.16, serum lactate 10.2 mmol/L. Urine ketones negative.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *Which metabolic emergency best explains these labs?* (`dka` / `hhs` / `mala` / `salicylate`) | `choice` | **`mala`** (`0.985`) | **`mala`** (`0.991`) |
| *Do the laboratory values support diabetic ketoacidosis?* | `noul` | `p(yes) = 0.086` | `p(yes) = 0.089` |

---

## 4. Abstention / Insufficient Information (Unbiopsied Lung Nodule)

**State:**
> 59yo M with 35 pack-year smoking history had a screening low-dose chest CT showing a 1.8 cm spiculated right upper lobe nodule. Pulmonology referral placed for PET-CT and biopsy. No pathology or mediastinal lymph node sampling has been performed yet.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *What is the confirmed TNM stage of lung cancer based on this note?* (`stage_i` / `stage_iii` / `stage_iv` / `insufficient_info`) | `choice` | **`insufficient_info`** (`0.968`) | **`insufficient_info`** (`0.990`) |

---

## 5. Numeric Lab Threshold (Stable CKD vs. KDIGO AKI)

**State:**
> 62yo M with stable stage 3b CKD (baseline creatinine 1.9 mg/dL over the past 18 months) seen in routine clinic for knee osteoarthritis. Labs today: Cr 2.0 mg/dL, BUN 28, K 4.3, urine output normal, no edema, no nephrotoxin exposure.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *Does this patient meet KDIGO criteria for acute kidney injury (rise >=0.3 mg/dL in 48h or >=1.5x baseline)?* | `noul` | `p(yes) = 0.087` | `p(yes) = 0.175` |

---

## 6. Clinical Trial Eligibility (Exclusion Trap)

**State:**
> Trial Criteria: Adults >=18y with symptomatic HFrEF (LVEF <=40%) on stable GDMT. Exclusions: eGFR <30 mL/min/1.73m2, active malignancy receiving chemotherapy, or pregnancy.
> Patient: 69yo M with NYHA III HFrEF (LVEF 30%) on carvedilol and sacubitril-valsartan, currently undergoing FOLFOX chemotherapy for stage III colon cancer, eGFR 28 mL/min/1.73m2.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *Is this patient eligible to enroll in the clinical trial?* | `noul` | `p(yes) = 0.019` | `p(yes) = 0.024` |

---

## 7. Minimal-Pair Sensitivity (Sepsis-3 Septic Shock Flip)

**Question (`noul`):** *Does the patient meet Sepsis-3 criteria for septic shock (vasopressor requirement and lactate > 2 mmol/L)?*

| Variant | Clinical State | Osler-9B | Osler-4B |
|---|---|---:|---:|
| **Positive** | *64yo F with E. coli bacteremia. Despite 30 mL/kg crystalloid, MAP remains 56 mmHg on norepinephrine 12 mcg/min. Repeat lactate 4.6 mmol/L.* | **`0.973`** | **`0.973`** |
| **Negative** | *64yo F with E. coli bacteremia. Following 30 mL/kg crystalloid, MAP improved to 78 mmHg with no vasopressors. Repeat lactate 1.2 mmol/L.* | **`0.042`** | **`0.037`** |

---

## 8. Drug–Drug Interaction & Ordinal `score` Behavior (Warfarin + TMP-SMX)

**State:**
> 78yo M on chronic warfarin 5 mg daily for atrial fibrillation (usual INR 2.3) was prescribed trimethoprim-sulfamethoxazole DS BID 5 days ago at urgent care for a UTI. Today he notices spontaneous ecchymoses on forearms and mild epistaxis; repeat INR is 6.4, Hgb stable at 12.8.

| Question | Type | Osler-9B | Osler-4B |
|---|---|---|---|
| *Is a clinically significant drug-drug interaction likely responsible for the supratherapeutic INR?* | `noul` | `p(yes) = 0.911` | `p(yes) = 0.907` |
| *Rate the urgency of holding warfarin and managing bleeding risk.* (`["none", "low", "moderate", "high", "emergent"]`) | `score` | `E = 1.87 / 4` (mode `2: moderate`) | `E = 2.43 / 4` (mode `2: moderate`) |

> **Practical tip on `score` rubrics:** Both models identify the CYP2C9 warfarin–TMP-SMX interaction cleanly (`~0.91`), but on unanchored single-word `score` scales (`"moderate"` vs. `"high"`), probabilities tend to cluster toward the middle. For production `score` tasks, explicitly anchor each level description with concrete clinical criteria (e.g., `"high: INR > 5 or active mucosal bleeding"`).
