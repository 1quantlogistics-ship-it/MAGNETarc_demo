"""
GPU-Accelerated Parallel Physics Engine for Batch Hull Simulation

This module implements vectorized physics calculations using PyTorch,
enabling simultaneous evaluation of multiple hull designs on GPU.

Key Features:
- Batch processing of 20-100+ designs simultaneously
- GPU acceleration via PyTorch tensors
- 10-50x speedup over sequential CPU evaluation
- Automatic fallback to CPU if GPU unavailable

Performance targets:
- 2x NVIDIA A40 (96GB total): 20-50 designs/batch, ~10-20 designs/sec throughput
- 1x A40 (48GB): 10-20 designs/batch
"""

from typing import List, Dict, Any, Optional
import math

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Parallel physics engine will use CPU fallback.")

from naval_domain.hull_parameters import HullParameters
from naval_domain.physics_engine import PhysicsEngine, PhysicsResults, WATER_DENSITY, GRAVITY, KINEMATIC_VISCOSITY, AIR_DENSITY, KNOTS_TO_MS


class ParallelPhysicsEngine:
    """
    GPU-accelerated batch physics engine.

    Processes multiple hull designs in parallel using PyTorch tensor operations.
    Automatically handles device selection (GPU/CPU) and memory management.
    """

    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        """
        Initialize parallel physics engine.

        Args:
            device: Target device ('cuda', 'cpu', or None for auto-detect)
            verbose: Enable verbose output
        """
        self.verbose = verbose

        if not TORCH_AVAILABLE:
            self.device = 'cpu'
            self.use_torch = False
            self.fallback_engine = PhysicsEngine(verbose=verbose)
            if verbose:
                print("PyTorch not available - using CPU fallback")
            return

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.use_torch = True

        if verbose:
            if self.device == 'cuda':
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("Using CPU (PyTorch available)")

    def simulate_batch(
        self,
        hull_params_list: List[Dict[str, Any]],
        return_dicts: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Simulate a batch of hull designs in parallel.

        Args:
            hull_params_list: List of hull parameter dictionaries
            return_dicts: If True, return list of dicts; if False, return PhysicsResults objects

        Returns:
            List of physics results (as dicts or PhysicsResults objects)

        Example:
            ```python
            engine = ParallelPhysicsEngine(device='cuda')

            designs = [
                {'length_overall': 18.0, 'beam': 2.0, ...},
                {'length_overall': 20.0, 'beam': 2.2, ...},
                ...
            ]

            results = engine.simulate_batch(designs)
            ```
        """
        if not self.use_torch:
            # Fallback to sequential CPU processing
            return self._fallback_batch_simulate(hull_params_list, return_dicts)

        n_designs = len(hull_params_list)

        if self.verbose:
            print(f"Simulating {n_designs} designs in parallel on {self.device}...")

        # Convert parameter dictionaries to HullParameters objects and validate
        hull_params_objects = []
        for params_dict in hull_params_list:
            try:
                hp = HullParameters(**params_dict)
                hp.validate()
                hull_params_objects.append(hp)
            except Exception as e:
                # For invalid designs, we'll mark them as failed later
                hull_params_objects.append(None)

        # Extract parameters into tensors (batch_size, 1)
        params_tensors = self._params_to_tensors(hull_params_objects)

        # Run batch calculations
        with torch.no_grad():  # No gradients needed for physics simulation
            results_tensors = self._batch_simulate_tensors(params_tensors, hull_params_objects)

        # Convert tensor results back to PhysicsResults objects
        results = self._tensors_to_results(results_tensors, hull_params_objects, hull_params_list)

        if return_dicts:
            return [r.to_dict() if r else None for r in results]
        else:
            return results

    def _fallback_batch_simulate(
        self,
        hull_params_list: List[Dict[str, Any]],
        return_dicts: bool
    ) -> List[Dict[str, Any]]:
        """Fallback to sequential CPU processing when PyTorch unavailable."""
        results = []

        for params_dict in hull_params_list:
            try:
                hp = HullParameters(**params_dict)
                result = self.fallback_engine.simulate(hp)
                results.append(result.to_dict() if return_dicts else result)
            except Exception as e:
                # Return placeholder for failed designs
                results.append(None)

        return results

    def _params_to_tensors(self, hull_params_objects: List[Optional[HullParameters]]) -> Dict[str, torch.Tensor]:
        """
        Convert list of HullParameters to batched tensors.

        Args:
            hull_params_objects: List of HullParameters (may contain None for invalid designs)

        Returns:
            Dictionary of parameter tensors, each of shape (batch_size,)
        """
        batch_size = len(hull_params_objects)

        # Initialize lists for each parameter
        param_names = [
            'length_overall', 'beam', 'hull_depth', 'hull_spacing',
            'deadrise_angle', 'freeboard', 'lcb_position',
            'prismatic_coefficient', 'waterline_beam', 'block_coefficient',
            'design_speed', 'displacement', 'draft'
        ]

        param_lists = {name: [] for name in param_names}

        # Extract parameters
        for hp in hull_params_objects:
            if hp is None:
                # Use dummy values for invalid designs
                for name in param_names:
                    param_lists[name].append(0.0)
            else:
                for name in param_names:
                    value = getattr(hp, name)
                    # Handle None draft
                    if value is None:
                        if name == 'draft':
                            # Estimate draft
                            value = hp.hull_depth * 0.5
                        else:
                            value = 0.0
                    param_lists[name].append(float(value))

        # Convert to tensors
        param_tensors = {
            name: torch.tensor(values, dtype=torch.float32, device=self.device)
            for name, values in param_lists.items()
        }

        return param_tensors

    def _batch_simulate_tensors(
        self,
        params: Dict[str, torch.Tensor],
        hull_params_objects: List[Optional[HullParameters]]
    ) -> Dict[str, torch.Tensor]:
        """
        Run batch physics calculations using tensors.

        Args:
            params: Dictionary of parameter tensors
            hull_params_objects: Original HullParameters objects (for reference)

        Returns:
            Dictionary of result tensors
        """
        # Extract parameters
        loa = params['length_overall']
        beam = params['beam']
        hull_depth = params['hull_depth']
        hull_spacing = params['hull_spacing']
        draft = params['draft']
        design_speed = params['design_speed']
        displacement = params['displacement']
        prismatic_coeff = params['prismatic_coefficient']
        block_coeff = params['block_coefficient']
        waterline_beam = params['waterline_beam']
        freeboard = params['freeboard']

        # === HYDROSTATIC CALCULATIONS ===
        displacement_volume = self._batch_displacement_volume(loa, beam, draft, block_coeff)
        displacement_mass = displacement_volume * WATER_DENSITY / 1000.0  # Convert to tons

        wetted_surface = self._batch_wetted_surface(loa, displacement_volume)

        # === STABILITY CALCULATIONS ===
        kb = self._batch_kb(draft, block_coeff)
        it = self._batch_transverse_inertia(loa, waterline_beam, prismatic_coeff, hull_spacing)
        gm = self._batch_metacentric_height(kb, it, displacement_volume, hull_depth)

        # === RESISTANCE CALCULATIONS ===
        speed_ms = design_speed * KNOTS_TO_MS

        fn = self._batch_froude_number(speed_ms, loa)
        rn = self._batch_reynolds_number(speed_ms, loa)

        rf = self._batch_frictional_resistance(speed_ms, wetted_surface, loa, rn)
        rr = self._batch_residuary_resistance(speed_ms, displacement_volume, fn, hull_spacing, loa, prismatic_coeff)
        ra = rf * 0.10  # 10% of friction for appendages
        r_air = self._batch_air_resistance(speed_ms, hull_spacing, freeboard)
        r_total = rf + rr + ra + r_air

        # === POWER CALCULATIONS ===
        pe = r_total * speed_ms / 1000.0  # kW
        pb = pe / 0.65  # Assume 65% propulsive efficiency
        power_per_ton = pb / displacement_mass

        # === PERFORMANCE SCORING ===
        stability_score = self._batch_stability_score(gm, hull_spacing, loa)
        speed_score = self._batch_speed_score(fn)
        efficiency_score = self._batch_efficiency_score(power_per_ton, displacement_mass)
        overall_score = 0.35 * stability_score + 0.35 * speed_score + 0.30 * efficiency_score

        return {
            'displacement_volume': displacement_volume,
            'displacement_mass': displacement_mass,
            'wetted_surface_area': wetted_surface,
            'draft_actual': draft,
            'metacentric_height': gm,
            'transverse_inertia': it,
            'volumetric_centroid': kb,
            'froude_number': fn,
            'reynolds_number': rn,
            'frictional_resistance': rf,
            'residuary_resistance': rr,
            'appendage_resistance': ra,
            'air_resistance': r_air,
            'total_resistance': r_total,
            'effective_power': pe,
            'brake_power': pb,
            'power_per_ton': power_per_ton,
            'stability_score': stability_score,
            'speed_score': speed_score,
            'efficiency_score': efficiency_score,
            'overall_score': overall_score,
        }

    # ========================================================================
    # BATCH HYDROSTATIC CALCULATIONS
    # ========================================================================

    def _batch_displacement_volume(
        self,
        loa: torch.Tensor,
        beam: torch.Tensor,
        draft: torch.Tensor,
        block_coeff: torch.Tensor
    ) -> torch.Tensor:
        """Batch calculation of displacement volume."""
        volume_per_hull = loa * beam * draft * block_coeff
        total_volume = 2.0 * volume_per_hull
        return total_volume

    def _batch_wetted_surface(
        self,
        loa: torch.Tensor,
        displacement_volume: torch.Tensor
    ) -> torch.Tensor:
        """Batch calculation of wetted surface area."""
        # S ≈ 3.8 × sqrt(∇ × L)
        wetted_surface = 3.8 * torch.sqrt(displacement_volume * loa)
        return wetted_surface

    # ========================================================================
    # BATCH STABILITY CALCULATIONS
    # ========================================================================

    def _batch_kb(self, draft: torch.Tensor, block_coeff: torch.Tensor) -> torch.Tensor:
        """Batch calculation of KB (center of buoyancy height)."""
        kb = draft * (0.45 + 0.05 * block_coeff)
        return kb

    def _batch_transverse_inertia(
        self,
        loa: torch.Tensor,
        waterline_beam: torch.Tensor,
        prismatic_coeff: torch.Tensor,
        hull_spacing: torch.Tensor
    ) -> torch.Tensor:
        """Batch calculation of transverse second moment."""
        # Waterline area per hull
        a_wl_per_hull = loa * waterline_beam * prismatic_coeff

        # Second moment about own centerline
        i_own = (waterline_beam ** 3 * loa * prismatic_coeff) / 12.0

        # Distance from centerline
        d = hull_spacing / 2.0

        # Parallel axis theorem
        i_t = 2.0 * (i_own + a_wl_per_hull * d**2)

        return i_t

    def _batch_metacentric_height(
        self,
        kb: torch.Tensor,
        it: torch.Tensor,
        displacement_volume: torch.Tensor,
        hull_depth: torch.Tensor
    ) -> torch.Tensor:
        """Batch calculation of GM."""
        bm = it / displacement_volume
        kg = hull_depth * 0.60  # Estimated CG height
        gm = kb + bm - kg
        return gm

    # ========================================================================
    # BATCH RESISTANCE CALCULATIONS
    # ========================================================================

    def _batch_froude_number(self, speed_ms: torch.Tensor, loa: torch.Tensor) -> torch.Tensor:
        """Batch Froude number calculation."""
        fn = speed_ms / torch.sqrt(GRAVITY * loa)
        return fn

    def _batch_reynolds_number(self, speed_ms: torch.Tensor, loa: torch.Tensor) -> torch.Tensor:
        """Batch Reynolds number calculation."""
        rn = speed_ms * loa / KINEMATIC_VISCOSITY
        return rn

    def _batch_frictional_resistance(
        self,
        speed_ms: torch.Tensor,
        wetted_surface: torch.Tensor,
        loa: torch.Tensor,
        rn: torch.Tensor
    ) -> torch.Tensor:
        """Batch frictional resistance using ITTC-1957."""
        # ITTC-1957 friction coefficient
        cf = 0.075 / (torch.log10(rn) - 2.0) ** 2

        # Form factor
        form_factor = 1.15

        # Frictional resistance
        rf = 0.5 * WATER_DENSITY * speed_ms**2 * wetted_surface * cf * form_factor

        return rf

    def _batch_residuary_resistance(
        self,
        speed_ms: torch.Tensor,
        displacement_volume: torch.Tensor,
        fn: torch.Tensor,
        hull_spacing: torch.Tensor,
        loa: torch.Tensor,
        prismatic_coeff: torch.Tensor
    ) -> torch.Tensor:
        """Batch residuary resistance calculation."""
        # Base coefficient (piecewise function)
        cr_base = torch.zeros_like(fn)

        # Fn < 0.35
        mask1 = fn < 0.35
        cr_base[mask1] = 0.0001 * fn[mask1]**3

        # 0.35 <= Fn < 0.50
        mask2 = (fn >= 0.35) & (fn < 0.50)
        cr_base[mask2] = 0.002 * (fn[mask2] - 0.35)**2 + 0.0001 * 0.35**3

        # Fn >= 0.50
        mask3 = fn >= 0.50
        cr_base[mask3] = 0.005 * (fn[mask3] - 0.50)**3 + 0.002 * 0.15**2 + 0.0001 * 0.35**3

        # Spacing factor
        spacing_ratio = hull_spacing / loa
        spacing_factor = torch.ones_like(spacing_ratio)
        spacing_factor[spacing_ratio < 0.25] = 1.4
        spacing_factor[(spacing_ratio >= 0.25) & (spacing_ratio < 0.35)] = 1.2

        # Prismatic coefficient factor
        cp_factor = 1.0 + (prismatic_coeff - 0.60) * 0.5

        # Combined coefficient
        cr = cr_base * spacing_factor * cp_factor

        # Residuary resistance
        rr = WATER_DENSITY * GRAVITY * displacement_volume * cr

        return rr

    def _batch_air_resistance(
        self,
        speed_ms: torch.Tensor,
        hull_spacing: torch.Tensor,
        freeboard: torch.Tensor
    ) -> torch.Tensor:
        """Batch air resistance calculation."""
        a_frontal = hull_spacing * freeboard * 0.7
        cd_air = 0.8
        r_air = 0.5 * AIR_DENSITY * speed_ms**2 * a_frontal * cd_air
        return r_air

    # ========================================================================
    # BATCH PERFORMANCE SCORING
    # ========================================================================

    def _batch_stability_score(
        self,
        gm: torch.Tensor,
        hull_spacing: torch.Tensor,
        loa: torch.Tensor
    ) -> torch.Tensor:
        """Batch stability scoring."""
        gm_score = torch.zeros_like(gm)

        # GM < 0: unstable
        mask1 = gm < 0
        gm_score[mask1] = 0.0

        # 0 <= GM < 0.35
        mask2 = (gm >= 0) & (gm < 0.35)
        gm_score[mask2] = 30.0 * (gm[mask2] / 0.35)

        # 0.35 <= GM < 1.0
        mask3 = (gm >= 0.35) & (gm < 1.0)
        gm_score[mask3] = 30.0 + 40.0 * ((gm[mask3] - 0.35) / 0.65)

        # 1.0 <= GM < 2.5
        mask4 = (gm >= 1.0) & (gm < 2.5)
        gm_score[mask4] = 70.0 + 30.0 * ((gm[mask4] - 1.0) / 1.5)

        # 2.5 <= GM < 5.0
        mask5 = (gm >= 2.5) & (gm < 5.0)
        gm_score[mask5] = 100.0 - 10.0 * ((gm[mask5] - 2.5) / 2.5)

        # GM >= 5.0
        mask6 = gm >= 5.0
        gm_score[mask6] = torch.clamp(90.0 - 5.0 * (gm[mask6] - 5.0), min=70.0)

        # Spacing bonus
        spacing_ratio = hull_spacing / loa
        spacing_bonus = torch.zeros_like(spacing_ratio)
        spacing_bonus[spacing_ratio > 0.35] = 10.0
        spacing_bonus[(spacing_ratio > 0.25) & (spacing_ratio <= 0.35)] = 5.0

        total_score = torch.clamp(gm_score + spacing_bonus, min=0.0, max=100.0)

        return total_score

    def _batch_speed_score(self, fn: torch.Tensor) -> torch.Tensor:
        """Batch speed scoring."""
        score = torch.zeros_like(fn)

        # Fn < 0.25
        mask1 = fn < 0.25
        score[mask1] = 50.0 + 50.0 * (fn[mask1] / 0.25)

        # 0.25 <= Fn < 0.50
        mask2 = (fn >= 0.25) & (fn < 0.50)
        score[mask2] = 100.0

        # 0.50 <= Fn < 0.80
        mask3 = (fn >= 0.50) & (fn < 0.80)
        score[mask3] = 100.0 - 20.0 * ((fn[mask3] - 0.50) / 0.30)

        # 0.80 <= Fn < 1.2
        mask4 = (fn >= 0.80) & (fn < 1.2)
        score[mask4] = 80.0 - 40.0 * ((fn[mask4] - 0.80) / 0.40)

        # Fn >= 1.2
        mask5 = fn >= 1.2
        deviation = torch.clamp((fn[mask5] - 1.2) / 0.5, max=1.0)
        score[mask5] = 40.0 - 30.0 * deviation

        return torch.clamp(score, min=0.0, max=100.0)

    def _batch_efficiency_score(
        self,
        power_per_ton: torch.Tensor,
        displacement: torch.Tensor
    ) -> torch.Tensor:
        """Batch efficiency scoring."""
        score = torch.zeros_like(power_per_ton)

        # < 5 kW/ton
        mask1 = power_per_ton < 5.0
        score[mask1] = 100.0

        # 5-10 kW/ton
        mask2 = (power_per_ton >= 5.0) & (power_per_ton < 10.0)
        score[mask2] = 100.0 - 20.0 * ((power_per_ton[mask2] - 5.0) / 5.0)

        # 10-20 kW/ton
        mask3 = (power_per_ton >= 10.0) & (power_per_ton < 20.0)
        score[mask3] = 80.0 - 40.0 * ((power_per_ton[mask3] - 10.0) / 10.0)

        # >= 20 kW/ton
        mask4 = power_per_ton >= 20.0
        deviation = torch.clamp((power_per_ton[mask4] - 20.0) / 20.0, max=1.0)
        score[mask4] = 40.0 - 40.0 * deviation

        # Bonus for larger vessels
        score[displacement > 50.0] += 5.0
        score[displacement > 100.0] += 5.0  # Additional 5 (10 total)

        return torch.clamp(score, min=0.0, max=100.0)

    def _tensors_to_results(
        self,
        results_tensors: Dict[str, torch.Tensor],
        hull_params_objects: List[Optional[HullParameters]],
        hull_params_dicts: List[Dict[str, Any]]
    ) -> List[PhysicsResults]:
        """
        Convert tensor results back to PhysicsResults objects.

        Args:
            results_tensors: Dictionary of result tensors
            hull_params_objects: Original HullParameters objects
            hull_params_dicts: Original parameter dictionaries

        Returns:
            List of PhysicsResults objects
        """
        results = []

        # Move tensors to CPU for conversion
        cpu_results = {k: v.cpu().numpy() for k, v in results_tensors.items()}

        for i, (hp, hp_dict) in enumerate(zip(hull_params_objects, hull_params_dicts)):
            if hp is None:
                # Invalid design
                results.append(None)
                continue

            # Extract values for this design
            result = PhysicsResults(
                hull_params=hp_dict,
                displacement_volume=float(cpu_results['displacement_volume'][i]),
                displacement_mass=float(cpu_results['displacement_mass'][i]),
                wetted_surface_area=float(cpu_results['wetted_surface_area'][i]),
                draft_actual=float(cpu_results['draft_actual'][i]),
                metacentric_height=float(cpu_results['metacentric_height'][i]),
                transverse_inertia=float(cpu_results['transverse_inertia'][i]),
                volumetric_centroid=float(cpu_results['volumetric_centroid'][i]),
                froude_number=float(cpu_results['froude_number'][i]),
                reynolds_number=float(cpu_results['reynolds_number'][i]),
                frictional_resistance=float(cpu_results['frictional_resistance'][i]),
                residuary_resistance=float(cpu_results['residuary_resistance'][i]),
                appendage_resistance=float(cpu_results['appendage_resistance'][i]),
                air_resistance=float(cpu_results['air_resistance'][i]),
                total_resistance=float(cpu_results['total_resistance'][i]),
                effective_power=float(cpu_results['effective_power'][i]),
                brake_power=float(cpu_results['brake_power'][i]),
                power_per_ton=float(cpu_results['power_per_ton'][i]),
                stability_score=float(cpu_results['stability_score'][i]),
                speed_score=float(cpu_results['speed_score'][i]),
                efficiency_score=float(cpu_results['efficiency_score'][i]),
                overall_score=float(cpu_results['overall_score'][i]),
                is_valid=True,
                failure_reasons=[],
            )

            results.append(result)

        return results


if __name__ == "__main__":
    # Demonstrate parallel physics engine
    from hull_parameters import get_baseline_catamaran
    import time

    print("=" * 70)
    print("PARALLEL PHYSICS ENGINE DEMONSTRATION")
    print("=" * 70)
    print()

    # Create batch of designs (variations on baseline)
    baseline = get_baseline_catamaran()
    baseline_dict = baseline.to_dict()

    # Generate 20 variations
    batch_size = 20
    designs = []

    for i in range(batch_size):
        design = baseline_dict.copy()
        # Vary length, beam, and hull spacing
        design['length_overall'] = 16.0 + i * 0.5
        design['beam'] = 1.8 + i * 0.05
        design['hull_spacing'] = 4.8 + i * 0.15
        design['design_speed'] = 20.0 + i * 1.0
        designs.append(design)

    # Initialize engines
    print(f"Testing batch of {batch_size} designs...")
    print()

    # GPU engine
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("GPU-Accelerated Batch Processing:")
        engine_gpu = ParallelPhysicsEngine(device='cuda', verbose=True)

        start = time.time()
        results_gpu = engine_gpu.simulate_batch(designs)
        elapsed_gpu = time.time() - start

        print(f"GPU Time: {elapsed_gpu:.3f}s ({batch_size/elapsed_gpu:.1f} designs/sec)")
        print(f"Average score: {sum(r['overall_score'] for r in results_gpu if r) / batch_size:.1f}/100")
        print()

    # CPU engine (PyTorch)
    print("CPU Batch Processing (PyTorch):")
    engine_cpu = ParallelPhysicsEngine(device='cpu', verbose=False)

    start = time.time()
    results_cpu = engine_cpu.simulate_batch(designs)
    elapsed_cpu = time.time() - start

    print(f"CPU (PyTorch) Time: {elapsed_cpu:.3f}s ({batch_size/elapsed_cpu:.1f} designs/sec)")
    print(f"Average score: {sum(r['overall_score'] for r in results_cpu if r) / batch_size:.1f}/100")
    print()

    # Sequential CPU engine (for comparison)
    print("Sequential CPU Processing (baseline):")
    sequential_engine = PhysicsEngine()

    start = time.time()
    results_seq = []
    for design in designs:
        hp = HullParameters(**design)
        result = sequential_engine.simulate(hp)
        results_seq.append(result.to_dict())
    elapsed_seq = time.time() - start

    print(f"Sequential Time: {elapsed_seq:.3f}s ({batch_size/elapsed_seq:.1f} designs/sec)")
    print(f"Average score: {sum(r['overall_score'] for r in results_seq) / batch_size:.1f}/100")
    print()

    # Speedup comparison
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Sequential CPU:  {elapsed_seq:.3f}s (1.00x baseline)")
    print(f"Parallel CPU:    {elapsed_cpu:.3f}s ({elapsed_seq/elapsed_cpu:.2f}x speedup)")

    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"Parallel GPU:    {elapsed_gpu:.3f}s ({elapsed_seq/elapsed_gpu:.2f}x speedup)")

    print()
    print("Sample results (first 3 designs):")
    for i in range(min(3, len(results_cpu))):
        r = results_cpu[i]
        print(f"  Design {i+1}: LOA={designs[i]['length_overall']:.1f}m, Score={r['overall_score']:.1f}/100")
