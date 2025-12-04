"""
Model download progress tracking for HuggingFace models
"""
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DownloadProgress:
    """Download progress information"""
    model_id: str
    is_downloading: bool
    total_size_bytes: int
    downloaded_bytes: int
    progress_percentage: float
    download_speed_mbps: float
    eta_seconds: Optional[int]
    files_total: int
    files_complete: int
    started_at: Optional[datetime]


class ModelDownloadTracker:
    """
    Track model download progress from HuggingFace cache
    """

    def __init__(self):
        self._downloads: Dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()
        self._monitoring_threads: Dict[str, threading.Thread] = {}

    def start_tracking(self, model_id: str, estimated_size_gb: float = 6.0):
        """
        Start tracking download progress for a model

        Args:
            model_id: HuggingFace model ID
            estimated_size_gb: Estimated total download size in GB
        """
        with self._lock:
            if model_id in self._downloads:
                return  # Already tracking

            # Initialize progress
            self._downloads[model_id] = DownloadProgress(
                model_id=model_id,
                is_downloading=True,
                total_size_bytes=int(estimated_size_gb * 1024 * 1024 * 1024),
                downloaded_bytes=0,
                progress_percentage=0.0,
                download_speed_mbps=0.0,
                eta_seconds=None,
                files_total=2,  # Most models have 2 safetensors files
                files_complete=0,
                started_at=datetime.utcnow(),
            )

            # Start monitoring thread
            thread = threading.Thread(
                target=self._monitor_download,
                args=(model_id,),
                daemon=True
            )
            thread.start()
            self._monitoring_threads[model_id] = thread

    def stop_tracking(self, model_id: str):
        """Stop tracking download progress"""
        with self._lock:
            if model_id in self._downloads:
                self._downloads[model_id].is_downloading = False

    def get_progress(self, model_id: str) -> Optional[DownloadProgress]:
        """Get current download progress"""
        with self._lock:
            return self._downloads.get(model_id)

    def is_downloading(self, model_id: str) -> bool:
        """Check if model is currently downloading"""
        with self._lock:
            if model_id not in self._downloads:
                return False
            return self._downloads[model_id].is_downloading

    def _monitor_download(self, model_id: str):
        """Monitor download progress in background thread"""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir_name = "models--" + model_id.replace("/", "--")
        model_cache_path = cache_dir / model_dir_name

        last_size = 0
        last_check_time = time.time()

        while True:
            with self._lock:
                if model_id not in self._downloads:
                    break
                if not self._downloads[model_id].is_downloading:
                    break

            try:
                # Calculate current cache size
                current_size = 0
                incomplete_files = 0
                complete_files = 0

                if model_cache_path.exists():
                    for file_path in model_cache_path.rglob("*"):
                        if file_path.is_file():
                            try:
                                size = file_path.stat().st_size
                                current_size += size

                                # Count incomplete files
                                if file_path.name.endswith('.incomplete'):
                                    incomplete_files += 1
                                elif 'blobs' in str(file_path) and not file_path.name.startswith('.'):
                                    complete_files += 1
                            except:
                                pass

                # Calculate speed
                current_time = time.time()
                time_delta = current_time - last_check_time
                size_delta = current_size - last_size

                speed_bps = size_delta / time_delta if time_delta > 0 else 0
                speed_mbps = speed_bps / (1024 * 1024)

                # Update progress
                with self._lock:
                    if model_id in self._downloads:
                        progress = self._downloads[model_id]
                        progress.downloaded_bytes = current_size
                        progress.download_speed_mbps = speed_mbps

                        # Calculate percentage
                        if progress.total_size_bytes > 0:
                            progress.progress_percentage = min(
                                (current_size / progress.total_size_bytes) * 100,
                                99.9  # Never show 100% until complete
                            )

                        # Estimate ETA
                        if speed_bps > 0:
                            remaining_bytes = progress.total_size_bytes - current_size
                            progress.eta_seconds = int(remaining_bytes / speed_bps)

                        # Update file counts
                        progress.files_complete = complete_files
                        progress.files_total = max(complete_files + incomplete_files, 2)

                        # Check if download is complete
                        if incomplete_files == 0 and current_size > 1024 * 1024 * 1024:  # > 1GB
                            # Has snapshots directory with files?
                            snapshots_dir = model_cache_path / "snapshots"
                            if snapshots_dir.exists():
                                snapshots = list(snapshots_dir.iterdir())
                                if snapshots and any(len(list(s.iterdir())) > 5 for s in snapshots if s.is_dir()):
                                    progress.is_downloading = False
                                    progress.progress_percentage = 100.0
                                    break

                last_size = current_size
                last_check_time = current_time

            except Exception as e:
                pass

            # Sleep before next check
            time.sleep(2)

    def to_dict(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Convert progress to dictionary for API response"""
        progress = self.get_progress(model_id)
        if not progress:
            return None

        result = {
            "model_id": progress.model_id,
            "is_downloading": progress.is_downloading,
            "total_size_bytes": progress.total_size_bytes,
            "total_size_gb": progress.total_size_bytes / (1024 ** 3),
            "downloaded_bytes": progress.downloaded_bytes,
            "downloaded_gb": progress.downloaded_bytes / (1024 ** 3),
            "progress_percentage": round(progress.progress_percentage, 1),
            "download_speed_mbps": round(progress.download_speed_mbps, 2),
            "eta_seconds": progress.eta_seconds,
            "files_total": progress.files_total,
            "files_complete": progress.files_complete,
            "started_at": progress.started_at.isoformat() if progress.started_at else None,
        }

        # Add human-readable ETA
        if progress.eta_seconds:
            if progress.eta_seconds < 60:
                result["eta_human"] = f"{progress.eta_seconds}s"
            elif progress.eta_seconds < 3600:
                minutes = progress.eta_seconds // 60
                result["eta_human"] = f"{minutes}m"
            else:
                hours = progress.eta_seconds // 3600
                minutes = (progress.eta_seconds % 3600) // 60
                result["eta_human"] = f"{hours}h {minutes}m"
        else:
            result["eta_human"] = "calculating..."

        return result


# Global tracker instance
_global_tracker: Optional[ModelDownloadTracker] = None


def get_download_tracker() -> ModelDownloadTracker:
    """Get or create global download tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ModelDownloadTracker()
    return _global_tracker
