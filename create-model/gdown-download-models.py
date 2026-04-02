import os
import zipfile
import shutil
import gdown
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)

SPECIALIST_DIR = BASE_DIR / "models" / "semeval_roberta_scanner_specialist"
if not SPECIALIST_DIR.exists():
    SPECIALIST_DIR.mkdir(parents=True)

TC_MODEL_PATH = BASE_DIR / "models" / "semeval_roberta_classifier"
if not TC_MODEL_PATH.exists():
    TC_MODEL_PATH.mkdir(parents=True)

google_drive_zip_ID = '1lLqG45VR24QxShlwAfkKB1vG9kHpUsx4'
google_drive_spec_zip_ID = '1O7UmT3L3qfILqdQavAqC7qxzeak86aGX'
google_drive_tc_context_zip_ID = '1PZXDKzsYcWGRE8U6nGSqq7dDi3F1TpSh'


def setup_models(file_id, target_path):
    target_path = Path(target_path).resolve()
    zip_temp = target_path.with_suffix(".zip")

    # Check if files already exist in the correct spot
    if (target_path / "model.safetensors").exists() or (target_path / "pytorch_model.bin").exists():
        print(f"Model weights detected locally at {target_path}")
        return True

    print(f"Model not found. Preparing {target_path}...")

    # Ensure the specific sub-folder exists
    target_path.mkdir(exist_ok=True, parents=True)

    url = f'https://drive.google.com/uc?id={file_id}'

    try:
        # 1. Download the zip
        gdown.download(url, str(zip_temp), quiet=False)

        # 2. Extract to a temporary location
        temp_extract = target_path / "temp_extraction"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        temp_extract.mkdir(parents=True)

        print("Unzipping and cleaning up structure...")
        with zipfile.ZipFile(zip_temp, 'r') as zip_ref:
            members = [m for m in zip_ref.namelist() if "__MACOSX" not in m]
            zip_ref.extractall(temp_extract, members=members)

        # 3. Move files from temp_extract into target_path
        for root, dirs, files in os.walk(temp_extract):
            for file in files:
                src_file = Path(root) / file
                dest_file = target_path / file
                shutil.move(str(src_file), str(dest_file))

        # 4. Final cleanup
        shutil.rmtree(temp_extract)
        if zip_temp.exists():
            os.remove(zip_temp)

        print(f"Model files are now in: {target_path}")
        return True

    except Exception as e:
        print(f"Error during setup: {e}")
        if zip_temp.exists():
            os.remove(zip_temp)
        return False


def main():
    base_model_exists = setup_models(google_drive_zip_ID, MODEL_DIR)
    specialist_model_exists = setup_models(google_drive_spec_zip_ID, SPECIALIST_DIR)
    tc_model_exists = setup_models(google_drive_tc_context_zip_ID, TC_MODEL_PATH)


if __name__ == "__main__":
    main()

