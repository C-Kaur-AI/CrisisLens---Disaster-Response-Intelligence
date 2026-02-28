"""
CrisisLens — Dataset Downloader
Downloads and prepares public crisis NLP datasets for training and evaluation.
"""

import os
import sys
import json
import zipfile
import tarfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


# Project root
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"


# ─── Dataset Registry ───

DATASETS = {
    "humaid": {
        "name": "HumAID — Human-Annotated Disaster Tweets",
        "description": "77K tweets annotated for 19 disaster events with humanitarian labels",
        "url": "https://crisisnlp.qcri.org/data/humaid/humaid_dataset.zip",
        "citation": "Alam et al., 2021 — HumAID: Human-Annotated Disaster Tweets",
        "license": "Research Use",
    },
    "crisisbench": {
        "name": "CrisisBench — Multi-source Crisis Dataset",
        "description": "170K+ tweets for crisis event classification benchmarking",
        "url": "https://github.com/firojalam/CrisisBench",
        "type": "github",
        "citation": "Alam et al., 2021 — CrisisBench",
        "license": "Research Use",
    },
}


# ─── Sample Data ───

SAMPLE_MESSAGES = [
    {
        "text": "URGENT: Family of 4 trapped on 2nd floor in Hatay district, water rising fast. Please send rescue team! #TurkeyEarthquake",
        "language": "en",
        "type": "RESCUE_REQUEST",
        "urgency": "CRITICAL",
    },
    {
        "text": "Necesitamos insulina urgente en el refugio de la escuela San Pedro. Hay 3 diabéticos sin medicamentos desde hace 2 días.",
        "language": "es",
        "type": "MEDICAL_EMERGENCY",
        "urgency": "CRITICAL",
    },
    {
        "text": "Main bridge on Highway 7 completely collapsed. Road to hospital blocked. No alternative route available.",
        "language": "en",
        "type": "INFRASTRUCTURE_DAMAGE",
        "urgency": "HIGH",
    },
    {
        "text": "दिल्ली में पुल टूट गया है, मुख्य सड़क पूरी तरह बंद है। कई गाड़ियां फंसी हैं।",
        "language": "hi",
        "type": "INFRASTRUCTURE_DAMAGE",
        "urgency": "HIGH",
    },
    {
        "text": "Le niveau d'eau monte rapidement dans le quartier Est de Lyon. Évacuation en cours.",
        "language": "fr",
        "type": "SITUATIONAL_UPDATE",
        "urgency": "MEDIUM",
    },
    {
        "text": "نحتاج ماء وطعام عاجل في مخيم الإيواء بمدينة حلب. أكثر من 200 عائلة بدون إمدادات",
        "language": "ar",
        "type": "SUPPLY_REQUEST",
        "urgency": "HIGH",
    },
    {
        "text": "300 families displaced from coastal area now sheltering in the local school gymnasium. Need blankets and food.",
        "language": "en",
        "type": "DISPLACEMENT",
        "urgency": "MEDIUM",
    },
    {
        "text": "I have a truck and supplies. Can volunteer to deliver food to affected areas in Antakya region.",
        "language": "en",
        "type": "VOLUNTEER_OFFER",
        "urgency": "LOW",
    },
    {
        "text": "At least 15 injured in building collapse at sector 7. Ambulances on scene but more needed.",
        "language": "en",
        "type": "CASUALTY_REPORT",
        "urgency": "HIGH",
    },
    {
        "text": "Weather forecast shows more rain expected tonight. Flood risk remains high for the next 48 hours.",
        "language": "en",
        "type": "SITUATIONAL_UPDATE",
        "urgency": "MEDIUM",
    },
    {
        "text": "Just had a great pizza at the new restaurant downtown. Best margherita ever! 🍕",
        "language": "en",
        "type": "NOT_CRISIS",
        "urgency": "NONE",
    },
    {
        "text": "Beautiful sunset from my balcony tonight. Nature is amazing! 🌅",
        "language": "en",
        "type": "NOT_CRISIS",
        "urgency": "NONE",
    },
]


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return dest


def create_sample_data():
    """Create sample data files for testing and demos."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    samples_json = SAMPLES_DIR / "sample_messages.json"
    with open(samples_json, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_MESSAGES, f, ensure_ascii=False, indent=2)
    print(f"✅ Sample messages saved to: {samples_json}")
    
    # Save as CSV for batch upload testing
    try:
        import pandas as pd
        df = pd.DataFrame(SAMPLE_MESSAGES)
        samples_csv = SAMPLES_DIR / "sample_messages.csv"
        df.to_csv(samples_csv, index=False, encoding="utf-8")
        print(f"✅ Sample CSV saved to: {samples_csv}")
    except ImportError:
        print("⚠️ pandas not installed, skipping CSV creation")


def download_fasttext_model():
    """Download the fastText language identification model."""
    model_dir = ROOT_DIR / "models"
    model_path = model_dir / "lid.176.bin"
    
    if model_path.exists():
        print(f"✅ FastText model already exists: {model_path}")
        return
    
    print("📥 Downloading fastText language identification model (126MB)...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    download_file(url, model_path, desc="lid.176.bin")
    print(f"✅ FastText model saved to: {model_path}")


def download_humaid():
    """Download HumAID dataset from Hugging Face for training/evaluation."""
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError:
        print("⚠️ Install: pip install huggingface_hub pyarrow")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    humaid_dir = RAW_DIR / "humaid"
    humaid_dir.mkdir(parents=True, exist_ok=True)

    if (humaid_dir / "train.csv").exists():
        print(f"✅ HumAID already exists: {humaid_dir}")
        return

    print("📥 Downloading HumAID from Hugging Face (QCRI/HumAID-all)...")
    # Use hf_hub_download to bypass dataset split metadata bug (expects 'dev' but has 'validation')
    repo_id = "QCRI/HumAID-all"
    files = {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    for split, fn in files.items():
        local = hf_hub_download(repo_id, fn, repo_type="dataset")
        pd.read_parquet(local).to_csv(humaid_dir / f"{split}.csv", index=False)
        print(f"   Saved {split}.csv")
    print(f"✅ HumAID saved to: {humaid_dir}")


def main():
    """Main download script."""
    print("=" * 60)
    print("🌍 CrisisLens — Data Setup")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n📝 Creating sample data...")
    create_sample_data()
    
    # Step 2: HumAID (optional, for train/evaluate)
    print("\n" + "-" * 40)
    r = input("Download HumAID dataset from Hugging Face (for train/evaluate)? [y/N]: ").strip().lower()
    if r == "y":
        download_humaid()

    # Step 3: fastText (optional)
    print("\n" + "-" * 40)
    response = input("Download fastText language model (126MB)? [y/N]: ").strip().lower()
    if response == "y":
        download_fasttext_model()
    else:
        print("⏭️ Skipping fastText download. Language detection will use langdetect fallback.")
    
    print("\n" + "=" * 60)
    print("✅ Data setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start API:       uvicorn src.api.main:app --reload")
    print("  2. Start Dashboard: streamlit run src/dashboard/app.py")
    print("  3. Run tests:       python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
