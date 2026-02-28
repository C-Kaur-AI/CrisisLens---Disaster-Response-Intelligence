"""
CrisisLens — User Guide content (embedded for the Streamlit User Guide tab).
"""

USER_GUIDE_MARKDOWN = r"""
# CrisisLens — Comprehensive User Guide & Technical Report

**Multilingual Crisis & Disaster Response NLP Pipeline**  
*Supporting UN Sustainable Development Goals #11 (Sustainable Cities) and #13 (Climate Action)*

---

## Executive Summary

**CrisisLens** is an AI-powered platform that processes social media and text messages during natural disasters to extract actionable intelligence for first responders, NGOs, and government agencies. It automatically classifies crisis relevance, detects event types, scores urgency, extracts locations, and identifies duplicate reports—in 176+ languages. This document explains every feature of the dashboard, why message analysis matters, and how the system works end-to-end.

---

## Why Analyze Crisis Messages?

### The Problem

During disasters (earthquakes, floods, hurricanes, conflicts), thousands of messages pour in via Twitter/X, WhatsApp, SMS, and emergency hotlines. Manual triage is impossible at scale. Responders need:

- **What** is happening (event type: rescue, medical, supply, etc.)
- **Where** it is (locations to deploy resources)
- **How urgent** (prioritize critical over low)
- **What's unique** (filter duplicates)

### The Solution

CrisisLens runs each message through an NLP pipeline that:

1. **Filters** — Keeps only crisis-relevant messages (ignores pizza reviews, memes, etc.)
2. **Classifies** — Labels event type (rescue, medical, supply, casualty, volunteer, etc.)
3. **Prioritizes** — Scores urgency (CRITICAL → HIGH → MEDIUM → LOW)
4. **Geocodes** — Extracts place names and resolves them to lat/lng
5. **Deduplicates** — Groups similar reports to reduce noise

This enables faster, data-driven response and better resource allocation.

---

## Dashboard Overview

The CrisisLens dashboard has several main tabs and a sidebar. Below is a full breakdown.

---

## Sidebar — Analysis Controls

| Element | Purpose |
|--------|---------|
| **Quick Samples** dropdown | Pre-loaded **INPUT** messages (raw text). The model receives these and outputs event type, urgency, locations. Includes explicit ("URGENT rescue needed") and implicit ("water at the door, kids with us, phone dying") examples. |
| **Session Stats** | Live counts: Total Processed, Relevant, Critical, Duplicates. |
| **Evaluation (HumAID)** | Benchmark metrics: Relevance F1, Type Macro-F1, Urgency κ. |
| **Limitations** | Known failure modes: implicit text, low-resource languages, obscure geocoding. |
| **Clear Results** button | Resets session stats and clears all analyzed results. Use this to start a fresh session. |

---

## Tab 1: Analyze

### Input Message (Left Column)

| Element | Purpose |
|--------|---------|
| **Text area** | Enter or paste a single message to analyze. Supports any language. You can also use text from the Quick Samples dropdown. |
| **Analyze** button | **Instant** for selected samples (pre-computed result). **Live** for custom text (runs the NLP pipeline; loads models on first use). |

### Analysis Result (Right Column)

For each analyzed message, the pipeline displays:

| Field | Meaning |
|-------|---------|
| **Crisis Related / Not Crisis Related** | Binary classification: is this message about a crisis or not? |
| **Language** | Detected language code (e.g. `en`, `es`) and confidence. |
| **Relevance** | Confidence score (0–100%) that the message is crisis-related. |
| **Event Types** | Labels such as RESCUE_REQUEST, MEDICAL_EMERGENCY, SUPPLY_REQUEST, INFRASTRUCTURE_DAMAGE, CASUALTY_REPORT, VOLUNTEER_OFFER, SITUATIONAL_UPDATE, DISPLACEMENT. |
| **Urgency** | CRITICAL, HIGH, MEDIUM, or LOW, plus a score. |
| **Locations** | Extracted place names (e.g. "Hatay", "San Pedro") with geocoded coordinates (latitude, longitude) when available. |
| **Duplicate** | Warning if this message is semantically similar to a previously analyzed one. |
| **Processed in Xms** | Processing time per message (or "Instant" for demo). |

---

## Tab 2: Priority Feed

Crisis messages ordered by urgency for responder triage: **CRITICAL → HIGH → MEDIUM → LOW**.

- Each message shows urgency, event types, language, and locations.
- CRITICAL items are expanded by default.
- **Export as CSV** — download the prioritized list for reporting or integration.

---

## Tab 3: Crisis Map

### What It Shows

An interactive map (Folium/OpenStreetMap) of all **crisis-related** messages that have **geocoded locations**.

### Map Features

| Element | Purpose |
|--------|---------|
| **Markers** | One per location. Color by urgency: Red (CRITICAL), Orange (HIGH), Beige (MEDIUM), Green (LOW). |
| **Marker icons** | Different icons for event types (life-ring, building, plus-sign, etc.). |
| **Popup** | Click a marker to see the message snippet, urgency, event type, and location. |
| **Legend** | Explains color coding: Critical, High, Medium, Low. |

### When the Map Is Empty

- No messages analyzed yet, or
- No crisis-related messages, or
- None of the crisis messages had extractable/geocodable locations.

Geocoding uses Nominatim (OpenStreetMap); obscure place names may not resolve.

---

## Tab 4: Analytics

### Summary Metrics (Top Row)

| Metric | Meaning |
|--------|---------|
| **Total Messages** | Number of messages analyzed in this session. |
| **Relevant** | Count and percentage of crisis-related messages. |
| **Critical** | Count of CRITICAL urgency messages. |
| **Duplicates** | Count of messages grouped as duplicates. |

### Charts

1. **Urgency Distribution** (Pie) — Proportions of CRITICAL, HIGH, MEDIUM, LOW among crisis messages.
2. **Event Type Distribution** (Bar) — Counts per event type (RESCUE_REQUEST, MEDICAL_EMERGENCY, etc.).
3. **Language Distribution** (Pie) — Counts per detected language.

### All Results Table

A table listing each analyzed message with: Relevant (Y/N), Urgency, Types, Language, Locations, Duplicate flag, Text snippet, and Processing time. **Export Full Results (CSV)** downloads the table.

---

## Pipeline Architecture (Under the Hood)

```
Raw Message
    ↓
Preprocess (clean, normalize)
    ↓
Language Detection (fastText or langdetect)
    ↓
Relevance Classification (BART / fine-tuned XLM-RoBERTa) — Is it crisis-related?
    ↓ (if relevant)
Type Classification (8 labels)
    ↓
Urgency Scoring (4 levels)
    ↓
GeoNER (XLM-RoBERTa) — Extract location entities
    ↓
Geocoding (Nominatim) — Resolve to lat/lng
    ↓
Semantic Deduplication — Group similar messages
    ↓
Dashboard / API / Alerts
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Relevance | Fine-tuned XLM-RoBERTa or facebook/bart-large-mnli |
| NER (locations) | Davlan/xlm-roberta-base-ner-hrl |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Language ID | langdetect (fallback when fastText unavailable) |
| Geocoding | Nominatim (OpenStreetMap) via geopy |
| Frontend | Streamlit, Folium, Plotly |
| Backend | FastAPI (optional) |

---

## Summary

CrisisLens turns unstructured crisis messages into structured, actionable intelligence. The **Analyze** tab handles single (and optionally batch) messages; the **Crisis Map** shows where events occur; the **Analytics** tab summarizes urgency, event types, and language. Together they support faster, more informed disaster response.

---

*Built for a more resilient world — Because every minute matters during a disaster.*
"""
