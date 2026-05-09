"""
KV Repository for RetrieveVGGT - Full History Frame Selection

Key features:
- Query-driven frame retrieval via descriptor similarity
- Segment Sampling for diverse coverage of high-similarity regions
- Pose-Aware KV Memory: spatial region classification based on position + viewing direction
- Pose-Aware KV Compression: per-region stride sampling
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class KVRepository:
    """
    Repository for storing all historical frames' KV states.
    
    Manages frame descriptors for similarity computation, KV state storage,
    and original frame ID tracking for ordering.
    """
    
    def __init__(
        self,
        num_layers: int = 24,
        top_k_frames: int = 46,
        anchor_frames: int = 1,
        device: torch.device = None,
        # Segment Sampling
        use_segment_sampling: bool = False,
        segment_max_ratio: float = 0.35,
        segment_min_gap: int = 3,
        segment_threshold_mode: str = 'mean+0.3std',  # 'mean', 'mean+0.3std', 'mean+0.6std'
        # Pose-Aware KV Memory (v3)
        use_pose_aware: bool = False,
        pos_cluster_radius: float = 1.0,       # BBox grid: 0=auto, >0=fixed grid_size
        grid_k: int = 3,                       # K: divide scene into K cells per axis
        n_dir_bins: int = 4,                   # D: azimuth direction bins
        # KV Compression (Pose-Aware per-region)
        use_kv_compression: bool = False,
        compression_interval: int = 200,
        compression_protect_recent: int = 50,
        compression_region_cap_ratio: float = 1.0,
        compression_delete_ratio: float = 0.5,
        compression_min_kv_ratio: float = 0.2,
        # Legacy params (accepted but ignored for backward compat)
        **kwargs,
    ):
        self.num_layers = num_layers
        self.top_k_frames = top_k_frames
        self.anchor_frames = anchor_frames
        self.device = device or torch.device('cuda')
        
        # Segment Sampling settings
        self.use_segment_sampling = use_segment_sampling
        self.segment_max_ratio = segment_max_ratio
        self.segment_min_gap = segment_min_gap
        self.segment_threshold_mode = segment_threshold_mode
        
        # Pose-Aware KV Memory settings (v3)
        self.use_pose_aware = use_pose_aware
        
        # BBox Grid region scheme:
        #   pos_cluster_radius <= 0 → auto: grid_size = scene_diameter / K
        #   pos_cluster_radius >  0 → fixed: grid_size = pos_cluster_radius
        self._grid_k = grid_k
        self._n_dir_bins = n_dir_bins
        self._cell_to_region: Dict[tuple, int] = {}
        self._region_to_cell: Dict[int, tuple] = {}
        self._next_region_id = 0
        self._grid_origin: Optional[torch.Tensor] = None
        self._grid_size: Optional[float] = None
        # Historical BBox: only grows, never shrinks (survives compression)
        self._historical_bbox_min: Optional[torch.Tensor] = None
        self._historical_bbox_max: Optional[torch.Tensor] = None
        
        if pos_cluster_radius <= 0:
            self._auto_grid = True
            self._auto_grid_warmup = 50
            self._auto_grid_update_interval = 100
            self._last_scene_diameter = 0.0
            self.pos_cluster_radius = float('inf')
        else:
            self._auto_grid = False
            self._grid_size = pos_cluster_radius
            self.pos_cluster_radius = pos_cluster_radius
        
        # KV Compression settings
        self.use_kv_compression = use_kv_compression
        self.compression_interval = compression_interval
        self.compression_protect_recent = compression_protect_recent
        self.compression_region_cap_ratio = compression_region_cap_ratio
        self.compression_delete_ratio = compression_delete_ratio
        self.compression_min_kv_ratio = compression_min_kv_ratio
        self._compression_count = 0
        self._total_frames_deleted = 0
        self._deleted_frames: set = set()
        
        # Pose storage
        self.frame_poses: List[torch.Tensor] = []       # [9] per frame
        self.frame_positions: List[torch.Tensor] = []   # [3] per frame
        self.frame_directions: List[torch.Tensor] = []  # [3] per frame
        
        # Spatial region management
        self.region_centers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.frame_region_ids: List[int] = []
        self.region_frame_lists: Dict[int, List[int]] = {}
        
        # KV storage: each layer stores list of (k, v) tuples
        self.kv_storage: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in range(num_layers)]
        self.frame_descriptors: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        
        # Original frame IDs (shared across layers)
        self.frame_ids: List[int] = []
        
        # Unified frame selection: Layer 0 computes, other layers reuse
        self._current_selected_indices: List[int] = []
        
        # Statistics
        self.total_frames = 0
        self.total_selections = 0
        
        # Log configuration
        if use_segment_sampling:
            logger.info(f"KVRepository: Segment Sampling enabled, max_ratio={segment_max_ratio}, threshold={segment_threshold_mode}")
        
        if use_pose_aware:
            if self._auto_grid:
                logger.info(f"KVRepository: Pose-Aware enabled (BBox Grid), auto grid_size=diameter/{self._grid_k}, "
                            f"warmup={self._auto_grid_warmup}, dir_bins={self._n_dir_bins}")
            else:
                logger.info(f"KVRepository: Pose-Aware enabled (BBox Grid), fixed grid_size={self._grid_size:.3f}, "
                            f"dir_bins={self._n_dir_bins}")
        
        if use_kv_compression:
            logger.info(f"KVRepository: KV Compression enabled, interval={compression_interval}, "
                        f"protect_recent={compression_protect_recent}, "
                        f"cap_ratio={compression_region_cap_ratio}, delete_ratio={compression_delete_ratio}")
        
    def reset(self):
        """Reset the repository for a new sequence."""
        self.kv_storage = [[] for _ in range(self.num_layers)]
        self.frame_descriptors = [[] for _ in range(self.num_layers)]
        self.frame_ids = []
        self._current_selected_indices = []
        self.total_frames = 0
        self.total_selections = 0
        
        # Reset Pose-Aware data
        self.frame_poses = []
        self.frame_positions = []
        self.frame_directions = []
        self.region_centers = []
        self.frame_region_ids = []
        self.region_frame_lists = {}
        # Reset BBox Grid state
        self._cell_to_region = {}
        self._region_to_cell = {}
        self._next_region_id = 0
        self._grid_origin = None
        self._grid_size = None if self._auto_grid else self._grid_size
        self._historical_bbox_min = None
        self._historical_bbox_max = None
        if hasattr(self, '_auto_grid') and self._auto_grid:
            self._last_scene_diameter = 0.0
            self.pos_cluster_radius = float('inf')
        
        # Reset compression state
        self._compression_count = 0
        self._total_frames_deleted = 0
        self._deleted_frames = set()
    
    def add_frame(
        self,
        layer_idx: int,
        k: torch.Tensor,  # [B, H, 1, tokens_per_frame, D]
        v: torch.Tensor,  # [B, H, 1, tokens_per_frame, D]
        frame_id: int,
        patch_start_idx: int = 0,
    ):
        """Add a new frame's KV to the repository."""
        # Record frame_id (only once, not per layer)
        if layer_idx == 0:
            self.frame_ids.append(frame_id)
            self.total_frames += 1
        
        # Store KV (squeeze the frame dimension)
        k_stored = k.squeeze(2).cpu()  # [B, H, tokens_per_frame, D]
        v_stored = v.squeeze(2).cpu()
        
        self.kv_storage[layer_idx].append((k_stored, v_stored))
        
        # Compute and store frame descriptor (mean of patch tokens)
        if patch_start_idx > 0 and patch_start_idx < k.shape[3]:
            descriptor = k[:, :, 0, patch_start_idx:, :].mean(dim=2)  # [B, H, D]
        else:
            descriptor = k[:, :, 0, :, :].mean(dim=2)  # [B, H, D]
        self.frame_descriptors[layer_idx].append(descriptor)
    
    def _get_frame_kv(self, layer_idx: int, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve KV states for a specific frame."""
        k_stored, v_stored = self.kv_storage[layer_idx][frame_idx]
        return k_stored.to(self.device), v_stored.to(self.device)
    
    def select_frames(
        self,
        layer_idx: int,
        query: torch.Tensor,  # [B, H, tokens_per_frame, D] or [B, H, D]
        patch_start_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Select frames from full history based on query relevance.
        
        UNIFIED SELECTION: Layer 0 computes the selection, Layer 1-23 reuse it.
        This ensures all layers attend to the same set of historical frames.
        
        Args:
            layer_idx: Which layer to select from
            query: Query states for the current frame
            patch_start_idx: Index where patch tokens start
            
        Returns:
            selected_k: [B, H, num_selected, tokens_per_frame, D]
            selected_v: [B, H, num_selected, tokens_per_frame, D]
            selected_frame_ids: List of original frame IDs that were selected
        """
        self.total_selections += 1
        num_frames = self.total_frames
        
        if num_frames == 0:
            return None, None, []
        
        max_frames = self.anchor_frames + self.top_k_frames
        
        # Unified selection: Layer 0 computes, layers 1-23 reuse
        if layer_idx == 0:
            _del = self._deleted_frames
            
            if num_frames <= max_frames:
                selected_indices = [i for i in range(num_frames) if i not in _del]
            else:
                anchor_indices = [i for i in range(min(self.anchor_frames, num_frames)) if i not in _del]
                
                candidate_start = self.anchor_frames
                candidate_end = num_frames
                
                if candidate_end > candidate_start:
                    selected_candidate_indices = self._select_with_decoder_k(
                        query, patch_start_idx, candidate_start, candidate_end,
                    )
                else:
                    selected_candidate_indices = []
                
                selected_indices = anchor_indices + selected_candidate_indices
            
            self._current_selected_indices = selected_indices
        else:
            selected_indices = self._current_selected_indices
        
        if not selected_indices:
            return None, None, []
        
        # Gather selected frames' KV
        selected_k_list = []
        selected_v_list = []
        
        for idx in selected_indices:
            k_sel, v_sel = self._get_frame_kv(layer_idx, idx)
            selected_k_list.append(k_sel)
            selected_v_list.append(v_sel)
        
        selected_k = torch.stack(selected_k_list, dim=2)
        selected_v = torch.stack(selected_v_list, dim=2)
        selected_frame_ids = [self.frame_ids[idx] for idx in selected_indices]
        
        return selected_k, selected_v, selected_frame_ids
    
    def _select_with_decoder_k(
        self,
        query: torch.Tensor,
        patch_start_idx: int,
        candidate_start: int,
        candidate_end: int,
    ) -> List[int]:
        """Select frames using decoder K features with raw dot product similarity."""
        # Compute query descriptor
        if query.dim() == 4:  # [B, H, tokens_per_frame, D]
            if patch_start_idx > 0 and patch_start_idx < query.shape[2]:
                query_descriptor = query[:, :, patch_start_idx:, :].mean(dim=2)  # [B, H, D]
            else:
                query_descriptor = query.mean(dim=2)
        else:  # Already [B, H, D]
            query_descriptor = query
        
        # Stack all frame descriptors: [num_frames, B, H, D]
        all_descriptors = torch.stack(self.frame_descriptors[0], dim=0)
        all_descriptors = all_descriptors.permute(1, 2, 0, 3)  # [B, H, num_frames, D]
        
        # Raw dot product similarity (Q·K)
        query_descriptor = query_descriptor.unsqueeze(2)  # [B, H, 1, D]
        similarity = torch.matmul(query_descriptor, all_descriptors.transpose(-1, -2))  # [B, H, 1, num_frames]
        similarity = similarity.squeeze(2)  # [B, H, num_frames]
        
        # Mask deleted frames to -inf
        if self._deleted_frames:
            for di in self._deleted_frames:
                if 0 <= di < similarity.shape[-1]:
                    similarity[:, :, di] = -float('inf')
        
        # Get candidate scores
        candidate_scores = similarity[:, :, candidate_start:candidate_end]
        avg_scores = candidate_scores.mean(dim=(0, 1))
        
        n_valid = int((avg_scores > -float('inf')).sum().item())
        k = min(self.top_k_frames, n_valid)
        
        if k == 0:
            return []
        
        if self.use_segment_sampling:
            sim_np = similarity.float().mean(dim=(0, 1)).cpu().numpy()
            selected = self._segment_sampling(sim_np, k, candidate_start, candidate_end)
            if self._deleted_frames:
                selected = [i for i in selected if i not in self._deleted_frames]
        else:
            _, top_indices = torch.topk(avg_scores, k=k)
            top_indices = top_indices.cpu().numpy().tolist()
            selected = sorted([candidate_start + i for i in top_indices])
        
        return selected
    
    def _segment_sampling(
        self,
        similarities: np.ndarray,
        topk: int,
        candidate_start: int,
        candidate_end: int,
    ) -> List[int]:
        """
        Segment Sampling: sparse sampling within high-score segments.
        
        Identifies contiguous high-score regions, allocates quota per segment,
        and samples sparsely within each to ensure diversity.
        Uses compact array (skipping deleted frames) to avoid segment fragmentation.
        """
        # Build compact array: skip deleted frames
        _del = self._deleted_frames
        if _del:
            valid_indices = [i for i in range(candidate_start, candidate_end) if i not in _del]
        else:
            valid_indices = list(range(candidate_start, candidate_end))
        
        n = len(valid_indices)
        if n <= topk:
            return valid_indices
        
        # Compact similarities (no -inf gaps, contiguous)
        compact_sims = similarities[valid_indices]
        
        # Step 1: Compute adaptive threshold
        mean_val = np.mean(compact_sims)
        std_val = np.std(compact_sims)
        
        if self.segment_threshold_mode == 'mean':
            threshold = mean_val
        elif self.segment_threshold_mode == 'mean+0.3std':
            threshold = mean_val + 0.3 * std_val
        elif self.segment_threshold_mode == 'mean+0.6std':
            threshold = mean_val + 0.6 * std_val
        else:
            threshold = mean_val
        
        # Step 2: Identify segments (in compact space)
        segments = []
        in_segment = False
        start = 0
        
        for i in range(n):
            if compact_sims[i] >= threshold:
                if not in_segment:
                    in_segment = True
                    start = i
            else:
                if in_segment:
                    in_segment = False
                    segments.append((start, i - 1))
        if in_segment:
            segments.append((start, n - 1))
        
        if not segments:
            top_compact = np.argsort(compact_sims)[-topk:]
            return sorted([valid_indices[ci] for ci in top_compact])
        
        # Step 3: Merge close segments
        merged = []
        for seg in segments:
            if merged and seg[0] - merged[-1][1] <= self.segment_min_gap:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        
        # Step 4: Compute segment properties and quotas
        segment_info = []
        for s, e in merged:
            seg_scores = compact_sims[s:e+1]
            peak_local = np.argmax(seg_scores)
            segment_info.append({
                'start': s,
                'end': e,
                'length': e - s + 1,
                'max_score': float(np.max(seg_scores)),
                'peak_idx': s + peak_local,
            })
        
        total_importance = sum(info['max_score'] for info in segment_info)
        
        if total_importance <= 0:
            top_compact = np.argsort(compact_sims)[-topk:]
            return sorted([valid_indices[ci] for ci in top_compact])
        
        max_per_segment = int(topk * self.segment_max_ratio)
        
        for info in segment_info:
            raw_quota = int(topk * info['max_score'] / total_importance)
            info['quota'] = min(raw_quota, max_per_segment, info['length'])
            info['quota'] = max(1, info['quota'])
        
        # Step 5: Sparse sampling within each segment
        def sample_from_segment(info):
            s, e = info['start'], info['end']
            length = info['length']
            quota = info['quota']
            peak = info['peak_idx']
            
            if quota >= length:
                return list(range(s, e + 1))
            
            if quota == 1:
                return [peak]
            
            if quota == 2:
                dist_to_start = peak - s
                dist_to_end = e - peak
                other = s if dist_to_start >= dist_to_end else e
                return sorted([peak, other])
            
            if quota == 3:
                return sorted(set([s, peak, e]))
            
            step = (length - 1) / (quota - 1)
            samples = [s + int(i * step) for i in range(quota)]
            
            if peak not in samples:
                closest_idx = min(range(len(samples)), 
                                 key=lambda i: abs(samples[i] - peak))
                samples[closest_idx] = peak
            
            return sorted(set(samples))
        
        all_samples = []
        for info in segment_info:
            samples = sample_from_segment(info)
            all_samples.extend(samples)
        
        all_samples = sorted(set(all_samples))
        
        # Step 6: Adjust to target count and map back to global indices
        if len(all_samples) >= topk:
            scored = [(valid_indices[ci], compact_sims[ci]) for ci in all_samples]
            scored.sort(key=lambda x: -x[1])
            selected = sorted([x[0] for x in scored[:topk]])
        else:
            selected_compact = set(all_samples)
            selected = set(valid_indices[ci] for ci in selected_compact)
            remaining = [(valid_indices[ci], compact_sims[ci]) for ci in range(n) 
                         if ci not in selected_compact]
            remaining.sort(key=lambda x: -x[1])
            
            for idx, score in remaining:
                if len(selected) >= topk:
                    break
                selected.add(idx)
            
            selected = sorted(selected)
        
        return selected[:topk]
    
    def _get_nearby_regions(self, query_region_id: int) -> set:
        """Get region IDs adjacent to the query region using BBox grid adjacency."""
        if query_region_id not in self._region_to_cell:
            return set()
        
        ix, iy, iz, dir_bin = self._region_to_cell[query_region_id]
        nearby = set()
        
        offsets = [(0,0,0), (1,0,0),(-1,0,0), (0,1,0),(0,-1,0), (0,0,1),(0,0,-1)]
        for dx, dy, dz in offsets:
            for d in range(self._n_dir_bins):
                cell = (ix+dx, iy+dy, iz+dz, d)
                if cell in self._cell_to_region:
                    rid = self._cell_to_region[cell]
                    if rid != query_region_id:
                        nearby.add(rid)
        
        return nearby
    
    # ==================== Pose-Aware Methods ====================
    
    @staticmethod
    def _quat_to_optical_axis(quat: torch.Tensor) -> torch.Tensor:
        """Extract optical axis (camera Z-axis) from quaternion."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        z_axis = torch.stack([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
        norm = z_axis.norm()
        if norm > 1e-6:
            return z_axis / norm
        return z_axis
    
    def update_pose(self, frame_id: int, camera_pose: torch.Tensor):
        """Store pose from CameraHead output. Called after each frame's inference."""
        pose_data = camera_pose.detach().cpu().squeeze()  # [9]
        self.frame_poses.append(pose_data)
        
        position = pose_data[:3].clone()
        self.frame_positions.append(position)
        
        # Update historical BBox (only grows, never shrinks)
        if self._historical_bbox_min is None:
            self._historical_bbox_min = position.clone()
            self._historical_bbox_max = position.clone()
        else:
            self._historical_bbox_min = torch.min(self._historical_bbox_min, position)
            self._historical_bbox_max = torch.max(self._historical_bbox_max, position)
        
        quat = pose_data[3:7]
        direction = self._quat_to_optical_axis(quat)
        self.frame_directions.append(direction)
        
        # Assign spatial region
        if self.use_pose_aware:
            self._assign_region(len(self.frame_poses) - 1)
        
        # Periodic KV compression check
        if (self.use_kv_compression and
            self.use_pose_aware and
            self.total_frames >= 200 and
            self.total_frames % self.compression_interval == 0):
            current_kv_count = len(self.kv_storage[0]) - len(self._deleted_frames) if self.kv_storage[0] else 0
            min_kv_count = int(self.total_frames * self.compression_min_kv_ratio)
            if current_kv_count >= min_kv_count:
                self._periodic_kv_compression()
            else:
                logger.info(
                    f"KV Compression skipped at frame {self.total_frames}: "
                    f"KV count {current_kv_count} < {self.total_frames}×{self.compression_min_kv_ratio} = {min_kv_count}"
                )
    
    def _quantize_direction(self, direction: torch.Tensor) -> int:
        """Quantize 3D viewing direction to one of n_dir_bins azimuth bins."""
        dx = direction[0].item()
        dz = direction[2].item()
        azimuth = math.atan2(dz, dx)
        if azimuth < 0:
            azimuth += 2 * math.pi
        bin_idx = int(azimuth / (2 * math.pi / self._n_dir_bins))
        return min(bin_idx, self._n_dir_bins - 1)
    
    def _pos_to_cell_ijk(self, pos: torch.Tensor) -> tuple:
        """Convert 3D position to grid cell indices (ix, iy, iz)."""
        if self._grid_origin is None or self._grid_size is None:
            return (0, 0, 0)
        offset = pos - self._grid_origin
        ix = int(math.floor(offset[0].item() / self._grid_size))
        iy = int(math.floor(offset[1].item() / self._grid_size))
        iz = int(math.floor(offset[2].item() / self._grid_size))
        return (ix, iy, iz)
    
    def _get_or_create_region(self, cell_key: tuple) -> int:
        """Get region_id for a cell key, creating a new region if needed."""
        if cell_key in self._cell_to_region:
            return self._cell_to_region[cell_key]
        
        region_id = self._next_region_id
        self._next_region_id += 1
        self._cell_to_region[cell_key] = region_id
        self._region_to_cell[region_id] = cell_key
        
        ix, iy, iz, dir_bin = cell_key
        if self._grid_origin is not None and self._grid_size is not None:
            pos_center = self._grid_origin + torch.tensor([
                (ix + 0.5) * self._grid_size,
                (iy + 0.5) * self._grid_size,
                (iz + 0.5) * self._grid_size,
            ])
        else:
            pos_center = self.frame_positions[0].clone() if self.frame_positions else torch.zeros(3)
        
        angle = (dir_bin + 0.5) * (2 * math.pi / self._n_dir_bins) - math.pi
        dir_center = torch.tensor([math.cos(angle), 0.0, math.sin(angle)])
        dir_center = F.normalize(dir_center, dim=0)
        
        self.region_centers.append((pos_center, dir_center))
        self.region_frame_lists[region_id] = []
        
        return region_id
    
    def _assign_region(self, frame_idx: int):
        """Assign a spatial region to a frame using BBox grid scheme."""
        if self._auto_grid:
            n_frames = len(self.frame_positions)
            need_regrid = False
            
            if n_frames == self._auto_grid_warmup:
                self._update_grid_params()
                need_regrid = True
            elif n_frames > self._auto_grid_warmup and n_frames % self._auto_grid_update_interval == 0:
                old_grid_size = self._grid_size
                self._update_grid_params()
                if old_grid_size and abs(self._grid_size - old_grid_size) / (old_grid_size + 1e-6) > 0.5:
                    need_regrid = True
            
            if need_regrid and n_frames > 1:
                self._regrid_all()
                return
        else:
            if self._grid_origin is None and len(self.frame_positions) > 0:
                self._grid_origin = self.frame_positions[0].clone()
        
        pos = self.frame_positions[frame_idx]
        dirn = self.frame_directions[frame_idx]
        
        ix, iy, iz = self._pos_to_cell_ijk(pos)
        dir_bin = self._quantize_direction(dirn)
        cell_key = (ix, iy, iz, dir_bin)
        
        region_id = self._get_or_create_region(cell_key)
        
        self.frame_region_ids.append(region_id)
        self.region_frame_lists[region_id].append(frame_idx)
    
    def _update_grid_params(self):
        """Compute BBox grid parameters from historical scene scale."""
        if self._historical_bbox_min is None or self._historical_bbox_max is None:
            return
        
        bbox_min = self._historical_bbox_min
        bbox_max = self._historical_bbox_max
        pos_range = bbox_max - bbox_min
        scene_diameter = pos_range.norm().item()
        
        if scene_diameter < 1e-6:
            return
        
        new_grid_size = max(scene_diameter / self._grid_k, 0.05)
        
        self._grid_origin = bbox_min.clone()
        self._grid_size = new_grid_size
        self._last_scene_diameter = scene_diameter
        self.pos_cluster_radius = new_grid_size
        
        logger.info(f"KVRepository: BBox grid updated: scene_diameter={scene_diameter:.3f}, "
                    f"grid_size={new_grid_size:.3f} (K={self._grid_k}), origin=[{bbox_min[0]:.3f},{bbox_min[1]:.3f},{bbox_min[2]:.3f}]")
    
    def _regrid_all(self):
        """Re-assign all frames to BBox grid cells."""
        self.region_centers = []
        self.frame_region_ids.clear()
        self.region_frame_lists.clear()
        self._cell_to_region.clear()
        self._region_to_cell.clear()
        self._next_region_id = 0
        
        for i in range(len(self.frame_positions)):
            pos = self.frame_positions[i]
            dirn = self.frame_directions[i]
            
            ix, iy, iz = self._pos_to_cell_ijk(pos)
            dir_bin = self._quantize_direction(dirn)
            cell_key = (ix, iy, iz, dir_bin)
            
            region_id = self._get_or_create_region(cell_key)
            self.frame_region_ids.append(region_id)
            self.region_frame_lists[region_id].append(i)
        
        logger.info(f"KVRepository: Re-gridded {len(self.frame_positions)} frames into "
                    f"{len(self.region_centers)} regions (grid_size={self._grid_size:.3f}, K={self._grid_k})")
    
    # ==================== KV Compression (Pose-Aware per-region) ====================
    
    def _periodic_kv_compression(self):
        """
        Pose-Aware KV Compression: per-region stride-sampling-based frame deletion.
        
        Strategy:
        1. Protect anchor frames and recent N frames
        2. Group remaining frames by BBox Grid region
        3. For over-cap regions: use stride sampling to keep evenly spaced frames
        4. Rebuild region_frame_lists index
        """
        num_frames = len(self.kv_storage[0])
        
        anchor_end = self.anchor_frames
        protect_start = max(anchor_end, num_frames - self.compression_protect_recent)
        
        if protect_start <= anchor_end:
            return
        
        compressible_indices = [i for i in range(anchor_end, protect_start) if i not in self._deleted_frames]
        if len(compressible_indices) < 10:
            return
        
        # Group by BBox region
        region_frames: Dict[int, List[int]] = {}
        for idx in compressible_indices:
            if idx < len(self.frame_region_ids):
                rid = self.frame_region_ids[idx]
                if rid not in region_frames:
                    region_frames[rid] = []
                region_frames[rid].append(idx)
        
        if not region_frames:
            return
        
        # Compute per-region cap
        n_compressible = len(compressible_indices)
        n_regions_with_frames = len(region_frames)
        avg_per_region = n_compressible / max(n_regions_with_frames, 1)
        cap = avg_per_region * self.compression_region_cap_ratio
        
        # Stride sampling for over-cap regions
        frames_to_keep = set()
        frames_to_keep.update(range(anchor_end))
        frames_to_keep.update(range(protect_start, num_frames))
        
        regions_compressed = 0
        
        for rid, frame_list in region_frames.items():
            if len(frame_list) <= int(cap):
                frames_to_keep.update(frame_list)
                continue
            
            n_keep = max(2, int(len(frame_list) * (1 - self.compression_delete_ratio)))
            kept = self._stride_sample_in_region(frame_list, n_keep)
            frames_to_keep.update(kept)
            regions_compressed += 1
        
        # Determine frames to delete
        all_compressible = set(compressible_indices) - self._deleted_frames
        frames_to_delete = sorted(all_compressible - frames_to_keep)
        
        if not frames_to_delete:
            return
        
        self._execute_deletion(frames_to_delete)
        
        self._compression_count += 1
        self._total_frames_deleted += len(frames_to_delete)
        active_kv = num_frames - len(self._deleted_frames)
        
        logger.info(
            f"KV Compression #{self._compression_count}: "
            f"active KV {active_kv}/{num_frames} "
            f"(tombstoned {len(frames_to_delete)} from {regions_compressed}/{n_regions_with_frames} regions, "
            f"cap={cap:.1f}, total_deleted_all_time={self._total_frames_deleted})"
        )
    
    def _stride_sample_in_region(self, frame_indices: List[int], n_keep: int) -> List[int]:
        """Stride Sampling: keep uniformly spaced frames by storage index order."""
        n = len(frame_indices)
        if n <= n_keep:
            return list(frame_indices)
        
        sorted_indices = sorted(frame_indices)
        
        if n_keep == 1:
            return [sorted_indices[n // 2]]
        
        step = (n - 1) / (n_keep - 1)
        kept = [sorted_indices[int(round(i * step))] for i in range(n_keep)]
        
        return list(dict.fromkeys(kept))
    
    def _execute_deletion(self, indices_to_delete: List[int]):
        """Tombstone deletion: null out KV tensors, keep metadata for indexing."""
        for idx in indices_to_delete:
            for layer_idx in range(self.num_layers):
                if idx < len(self.kv_storage[layer_idx]):
                    self.kv_storage[layer_idx][idx] = None
            self._deleted_frames.add(idx)
    
    # ==================== Statistics ====================
    
    def get_pose_stats(self) -> dict:
        """Get pose-aware statistics for debugging and visualization."""
        if not self.use_pose_aware or len(self.frame_poses) == 0:
            return {}
        
        stats = {
            "total_frames_with_pose": len(self.frame_poses),
            "total_regions": len(self.region_centers),
            "region_sizes": {rid: len(frames) for rid, frames in self.region_frame_lists.items()},
            "grid_size": self._grid_size if self._grid_size else "warmup",
            "grid_k": self._grid_k,
            "n_dir_bins": self._n_dir_bins,
        }
        
        if hasattr(self, '_auto_grid') and self._auto_grid:
            stats["auto_grid"] = True
            stats["scene_diameter"] = self._last_scene_diameter
        
        if len(self.frame_positions) > 0:
            pos_stack = torch.stack(self.frame_positions)
            stats["pos_range"] = {
                "min": pos_stack.min(dim=0).values.tolist(),
                "max": pos_stack.max(dim=0).values.tolist(),
                "mean": pos_stack.mean(dim=0).tolist(),
            }
        
        if self.use_kv_compression:
            total_slots = len(self.kv_storage[0]) if self.kv_storage[0] else 0
            stats["compression"] = {
                "enabled": True,
                "interval": self.compression_interval,
                "min_kv_ratio": self.compression_min_kv_ratio,
                "compressions_done": self._compression_count,
                "total_frames_deleted": self._total_frames_deleted,
                "active_kv_count": total_slots - len(self._deleted_frames),
                "tombstoned_count": len(self._deleted_frames),
            }
        
        return stats
    
    def get_stats(self) -> dict:
        """Get repository statistics."""
        total_slots = len(self.kv_storage[0]) if self.kv_storage[0] else 0
        return {
            "total_frames": self.total_frames,
            "total_selections": self.total_selections,
            "memory_frames": total_slots - len(self._deleted_frames),
        }
