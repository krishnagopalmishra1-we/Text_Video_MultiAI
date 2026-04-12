"""Google Cloud Storage integration for video output uploads."""
import os
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

logger = logging.getLogger(__name__)


class VideoStorage:
    """Upload and manage video artifacts in GCS with signed URLs."""
    
    def __init__(self, bucket_name: str = "videogen-outputs"):
        """Initialize GCS storage client."""
        if not HAS_GCS:
            logger.warning("google-cloud-storage not installed. GCS uploads disabled.")
            self.client = None
            self.bucket = None
            return
        
        try:
            self.client = gcs.Client()
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Initialized GCS storage: gs://{bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS: {e}")
            self.client = None
            self.bucket = None
    
    def upload_file(self, local_path: Path, job_id: str, prefix: str = "jobs") -> Optional[str]:
        """Upload file to GCS and return public URL."""
        if not self.bucket:
            logger.warning(f"GCS not available, skipping upload of {local_path}")
            return None
        
        try:
            blob_path = f"{prefix}/{job_id}/{local_path.name}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path.name} to gs://{self.bucket.name}/{blob_path}")
            return f"gs://{self.bucket.name}/{blob_path}"
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to GCS: {e}")
            return None
    
    def generate_signed_url(
        self, 
        blob_path: str, 
        expiry_hours: int = 24
    ) -> Optional[str]:
        """Generate signed URL for file download (valid for expiry_hours)."""
        if not self.bucket:
            logger.warning("GCS not available, cannot generate signed URL")
            return None
        
        try:
            blob = self.bucket.blob(blob_path)
            url = blob.generate_signed_url(
                expiration=timedelta(hours=expiry_hours),
                method="GET"
            )
            logger.info(f"Generated signed URL (expires in {expiry_hours}h): {blob_path[:50]}...")
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {blob_path}: {e}")
            return None
    
    def upload_video(self, video_path: Path, job_id: str) -> Optional[str]:
        """Upload final video and return signed download URL."""
        gcs_path = self.upload_file(video_path, job_id, prefix="videos")
        if not gcs_path:
            return None
        
        # Extract blob path from gs:// URI
        blob_path = gcs_path.replace(f"gs://{self.bucket.name}/", "")
        signed_url = self.generate_signed_url(blob_path, expiry_hours=24)
        return signed_url
    
    def list_job_outputs(self, job_id: str) -> list:
        """List all files for a given job."""
        if not self.bucket:
            return []
        
        try:
            blobs = list(self.bucket.list_blobs(prefix=f"jobs/{job_id}/"))
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list job outputs: {e}")
            return []


# Singleton instance
_storage: Optional[VideoStorage] = None


def get_storage() -> VideoStorage:
    """Get or create singleton storage instance."""
    global _storage
    if _storage is None:
        bucket_name = os.environ.get("GCS_BUCKET", "videogen-outputs")
        _storage = VideoStorage(bucket_name)
    return _storage
