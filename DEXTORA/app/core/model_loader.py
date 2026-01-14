import os
from google.cloud import storage
from app.core.config import settings

def download_models_from_gcs():
    """
    Downloads model artifacts from GCS bucket to local path on startup.
    This ensures ephemeral containers always have the latest 'global brain'.
    """
    bucket_name = settings.GCS_BUCKET_NAME
    if not bucket_name:
        print("⚠️ GCS_BUCKET_NAME not set. Skipping GCS download (assuming local development).")
        return

    print(f"⬇️ Downloading models from GCS Bucket: {bucket_name}...")
    
    # 1. Initialize Client (Picks up credentials from Environment automatically)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}")
        return

    # 2. Define artifacts to fetch
    # Map GCS Filename -> Local Destination
    artifacts = {
        "saint_weights.pt": "app/ml_assets/saint_weights.pt",
        "ppo_student_policy.zip": "app/ml_assets/ppo_student_policy.zip"
    }

    prefix = settings.GCS_MODEL_PREFIX
    
    for gcs_filename, local_path in artifacts.items():
        # Handle prefix if models are in a subfolder in the bucket
        blob_name = f"{prefix}/{gcs_filename}" if prefix else gcs_filename
        # Remove double slashes if prefix was empty but had trailing slash or handled poorly
        blob_name = blob_name.replace("//", "/")
        
        blob = bucket.blob(blob_name)
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            print(f"   - Fetching {blob_name} -> {local_path}")
            blob.download_to_filename(local_path)
            print(f"     ✅ Downloaded.")
        except Exception as e:
            print(f"     ❌ Error downloading {blob_name}: {e}")
            # We don't raise here to allow partial success, but in prod you might want to crash.
    
    print("✅ Model download sequence complete.")
