"""
3D Mesh Comparison Metrics

Metrics for comparing reconstructed 3D models against ground truth:
- Chamfer Distance: Average nearest-neighbor distance
- Hausdorff Distance: Maximum deviation
- IoU (Intersection over Union): Volume overlap
- Surface Distance: Point-to-surface distance
- F-Score: Precision/recall at distance threshold
"""
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two 3D models"""
    # Distance metrics (lower is better)
    chamfer_distance: float
    hausdorff_distance: float
    mean_surface_distance: float

    # Overlap metrics (higher is better, 0-1)
    iou_3d: float  # Intersection over Union
    f_score: float  # F-score at threshold

    # Additional stats
    precision: float  # Points in prediction close to ground truth
    recall: float  # Points in ground truth close to prediction

    # Metadata
    num_points_pred: int
    num_points_gt: int
    threshold: float  # Distance threshold used for F-score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chamfer_distance": self.chamfer_distance,
            "hausdorff_distance": self.hausdorff_distance,
            "mean_surface_distance": self.mean_surface_distance,
            "iou_3d": self.iou_3d,
            "f_score": self.f_score,
            "precision": self.precision,
            "recall": self.recall,
            "num_points_pred": self.num_points_pred,
            "num_points_gt": self.num_points_gt,
            "threshold": self.threshold,
        }

    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Comparison Results:\n"
            f"  Chamfer Distance:  {self.chamfer_distance:.4f}\n"
            f"  Hausdorff Distance: {self.hausdorff_distance:.4f}\n"
            f"  Mean Surface Dist:  {self.mean_surface_distance:.4f}\n"
            f"  IoU (3D):          {self.iou_3d:.4f} ({self.iou_3d*100:.1f}%)\n"
            f"  F-Score:           {self.f_score:.4f} ({self.f_score*100:.1f}%)\n"
            f"  Precision:         {self.precision:.4f}\n"
            f"  Recall:            {self.recall:.4f}"
        )


class MeshComparison:
    """
    Compare two 3D meshes and compute various metrics.

    Used for evaluating reconstruction quality against ground truth.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        threshold: float = 0.05,  # 5cm default threshold
        voxel_resolution: int = 64,
    ):
        """
        Args:
            num_samples: Number of points to sample from mesh surfaces
            threshold: Distance threshold for F-score (meters)
            voxel_resolution: Resolution for voxelization (IoU calculation)
        """
        self.num_samples = num_samples
        self.threshold = threshold
        self.voxel_resolution = voxel_resolution

    def compare(self, pred_mesh, gt_mesh) -> ComparisonResult:
        """
        Compare predicted mesh against ground truth.

        Args:
            pred_mesh: Predicted/reconstructed mesh (trimesh.Trimesh)
            gt_mesh: Ground truth mesh (trimesh.Trimesh)

        Returns:
            ComparisonResult with all metrics
        """
        import trimesh

        # Sample points from surfaces
        pred_points = self._sample_surface(pred_mesh)
        gt_points = self._sample_surface(gt_mesh)

        if len(pred_points) == 0 or len(gt_points) == 0:
            logger.warning("Empty point cloud, returning zero metrics")
            return ComparisonResult(
                chamfer_distance=float('inf'),
                hausdorff_distance=float('inf'),
                mean_surface_distance=float('inf'),
                iou_3d=0.0,
                f_score=0.0,
                precision=0.0,
                recall=0.0,
                num_points_pred=len(pred_points),
                num_points_gt=len(gt_points),
                threshold=self.threshold,
            )

        # Compute distance metrics
        chamfer = self._chamfer_distance(pred_points, gt_points)
        hausdorff = self._hausdorff_distance(pred_points, gt_points)
        mean_surface = self._mean_surface_distance(pred_points, gt_points)

        # Compute overlap metrics
        precision, recall = self._precision_recall(pred_points, gt_points, self.threshold)
        f_score = self._f_score(precision, recall)

        # Compute IoU via voxelization
        iou = self._iou_3d(pred_mesh, gt_mesh)

        return ComparisonResult(
            chamfer_distance=chamfer,
            hausdorff_distance=hausdorff,
            mean_surface_distance=mean_surface,
            iou_3d=iou,
            f_score=f_score,
            precision=precision,
            recall=recall,
            num_points_pred=len(pred_points),
            num_points_gt=len(gt_points),
            threshold=self.threshold,
        )

    def _sample_surface(self, mesh, num_samples: Optional[int] = None) -> np.ndarray:
        """Sample points uniformly from mesh surface"""
        if num_samples is None:
            num_samples = self.num_samples

        try:
            # trimesh's sample method
            points, _ = mesh.sample(num_samples, return_index=True)
            return np.array(points)
        except Exception as e:
            logger.warning(f"Surface sampling failed: {e}")
            # Fallback to vertices
            if len(mesh.vertices) > num_samples:
                indices = np.random.choice(len(mesh.vertices), num_samples, replace=False)
                return mesh.vertices[indices]
            return mesh.vertices

    def _chamfer_distance(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        """
        Compute Chamfer Distance between two point clouds.

        CD = mean(min_b ||a - b||) + mean(min_a ||b - a||)
        """
        from scipy.spatial import cKDTree

        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)

        # Distance from A to nearest in B
        dist_a_to_b, _ = tree_b.query(points_a, k=1)
        # Distance from B to nearest in A
        dist_b_to_a, _ = tree_a.query(points_b, k=1)

        chamfer = np.mean(dist_a_to_b) + np.mean(dist_b_to_a)
        return float(chamfer)

    def _hausdorff_distance(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        """
        Compute Hausdorff Distance (maximum deviation).

        HD = max(max(min_b ||a - b||), max(min_a ||b - a||))
        """
        from scipy.spatial import cKDTree

        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)

        dist_a_to_b, _ = tree_b.query(points_a, k=1)
        dist_b_to_a, _ = tree_a.query(points_b, k=1)

        hausdorff = max(np.max(dist_a_to_b), np.max(dist_b_to_a))
        return float(hausdorff)

    def _mean_surface_distance(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        """
        Compute mean surface distance (one-way, A to B).
        """
        from scipy.spatial import cKDTree

        tree_b = cKDTree(points_b)
        dist_a_to_b, _ = tree_b.query(points_a, k=1)

        return float(np.mean(dist_a_to_b))

    def _precision_recall(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float]:
        """
        Compute precision and recall at distance threshold.

        Precision: fraction of predicted points within threshold of GT
        Recall: fraction of GT points within threshold of prediction
        """
        from scipy.spatial import cKDTree

        tree_pred = cKDTree(pred_points)
        tree_gt = cKDTree(gt_points)

        # Precision: pred -> gt
        dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        precision = np.mean(dist_pred_to_gt < threshold)

        # Recall: gt -> pred
        dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
        recall = np.mean(dist_gt_to_pred < threshold)

        return float(precision), float(recall)

    def _f_score(self, precision: float, recall: float) -> float:
        """Compute F-score from precision and recall"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _iou_3d(self, pred_mesh, gt_mesh) -> float:
        """
        Compute 3D IoU via voxelization.

        IoU = |intersection| / |union|
        """
        try:
            # Voxelize both meshes
            pred_voxels = self._voxelize(pred_mesh)
            gt_voxels = self._voxelize(gt_mesh)

            if pred_voxels is None or gt_voxels is None:
                return 0.0

            # Compute intersection and union
            intersection = np.logical_and(pred_voxels, gt_voxels).sum()
            union = np.logical_or(pred_voxels, gt_voxels).sum()

            if union == 0:
                return 0.0

            return float(intersection / union)

        except Exception as e:
            logger.warning(f"IoU calculation failed: {e}")
            return 0.0

    def _voxelize(self, mesh) -> Optional[np.ndarray]:
        """Convert mesh to voxel grid"""
        try:
            # Get combined bounds for consistent voxelization
            bounds = mesh.bounds
            extent = bounds[1] - bounds[0]
            max_extent = max(extent)

            pitch = max_extent / self.voxel_resolution

            # Create voxel grid
            voxels = mesh.voxelized(pitch=pitch)

            return voxels.matrix

        except Exception as e:
            logger.warning(f"Voxelization failed: {e}")
            return None


def compute_metrics(pred_mesh, gt_mesh, **kwargs) -> ComparisonResult:
    """
    Convenience function to compute all comparison metrics.

    Args:
        pred_mesh: Predicted mesh (trimesh.Trimesh)
        gt_mesh: Ground truth mesh (trimesh.Trimesh)
        **kwargs: Additional arguments for MeshComparison

    Returns:
        ComparisonResult with all metrics
    """
    comparator = MeshComparison(**kwargs)
    return comparator.compare(pred_mesh, gt_mesh)


def compute_2d_metrics(pred_image: np.ndarray, gt_image: np.ndarray) -> Dict[str, float]:
    """
    Compute 2D image comparison metrics (for floor plan comparison).

    Args:
        pred_image: Predicted floor plan (2D array)
        gt_image: Ground truth floor plan (2D array)

    Returns:
        Dictionary with 2D metrics
    """
    # Ensure same size
    if pred_image.shape != gt_image.shape:
        from PIL import Image
        pred_pil = Image.fromarray(pred_image)
        pred_pil = pred_pil.resize((gt_image.shape[1], gt_image.shape[0]))
        pred_image = np.array(pred_pil)

    # Binarize (assuming white background, black lines)
    pred_binary = pred_image < 128
    gt_binary = gt_image < 128

    # IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    # Precision/Recall for lines
    true_pos = intersection
    pred_pos = pred_binary.sum()
    gt_pos = gt_binary.sum()

    precision = true_pos / pred_pos if pred_pos > 0 else 0.0
    recall = true_pos / gt_pos if gt_pos > 0 else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MSE
    mse = np.mean((pred_image.astype(float) - gt_image.astype(float)) ** 2)

    # SSIM (if scikit-image available)
    ssim = 0.0
    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(pred_image, gt_image)
    except ImportError:
        pass

    return {
        "iou_2d": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f_score": float(f_score),
        "mse": float(mse),
        "ssim": float(ssim),
    }
