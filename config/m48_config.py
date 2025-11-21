"""
M48 Mission Configuration for NAVSEA HC-MASC Program

Configures the MAGNET autonomous research system specifically for
Magnet Defense M48 High-Capacity Modular Attack Surface Craft optimization.

Mission Context:
- Program: NAVSEA MASC N00024-25-R-6314
- Platform: 48-meter twin-hull catamaran (proven 32,000 NM sea trials)
- Objective: Replace Independence-class LCS missions on smaller platform
- Requirements: Distributed operations, ISR, ASW/MCM, missile tracking

Research Focus:
- Multi-objective optimization: Stability × Range × Payload × Speed
- Mission-specific variants: ISR, ASW/MCM, picket ship, logistics
- Pareto frontier exploration for Navy proposal substantiation
- Sea State 9 survivability validation
- RCS reduction vs. performance tradeoffs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json


@dataclass
class M48MissionConfig:
    """
    M48-specific mission configuration for autonomous research.

    Defines:
    - NAVSEA HC-MASC program requirements
    - Mission profiles (ISR, ASW/MCM, picket, logistics)
    - Optimization objectives and constraints
    - Research questions for autonomous exploration
    """

    # === PROGRAM IDENTIFICATION ===
    program_name: str = "NAVSEA HC-MASC"
    platform_name: str = "Magnet Defense M48"
    solicitation_number: str = "N00024-25-R-6314"
    vessel_class: str = "Medium Unmanned Surface Vehicle (mUSV)"

    # === MISSION PROFILES ===
    primary_missions: List[str] = field(default_factory=lambda: [
        "ISR",  # Intelligence, Surveillance, Reconnaissance
        "COMMUNICATIONS",  # Data relay, MUOS connectivity
        "SURFACE_WARFARE",  # Missile tracking, defense sensors
        "ASW",  # Anti-submarine warfare
        "MCM",  # Mine countermeasures
        "PICKET_SHIP",  # Distributed sensor node
        "LOGISTICS",  # Contested resupply
    ])

    secondary_missions: List[str] = field(default_factory=lambda: [
        "EW_PASSIVE",  # Electronic warfare (passive sensing)
        "UAV_MOTHERSHIP",  # UAV/UUV/USV launch platform
        "ESCORT",  # Force protection
    ])

    # === PLATFORM REQUIREMENTS (from NAVSEA solicitation) ===
    threshold_requirements: Dict[str, Any] = field(default_factory=lambda: {
        'payload_capacity': 144.0,  # Metric tons (4 × 36-ton containers)
        'min_range': 8000.0,  # NM (with max payload)
        'max_speed_threshold': 24.0,  # Knots (with max payload)
        'sea_state': 9,  # WMO Sea State survivability
        'autonomy_level': 'TRL-9',  # L3Harris ASView
        'modular_payload': True,  # Containerized mission modules
    })

    objective_requirements: Dict[str, Any] = field(default_factory=lambda: {
        'payload_capacity': 200.0,  # Metric tons (stretch goal)
        'min_range': 15000.0,  # NM (proven M48 performance)
        'max_speed_objective': 30.0,  # Knots (proven M48 capability)
        'rcs_reduction': True,  # Faceted geometry
        'collaborative_autonomy': True,  # AMORPHOUS (L3Harris)
    })

    # === OPTIMIZATION OBJECTIVES ===
    optimization_weights: Dict[str, float] = field(default_factory=lambda: {
        'stability': 0.40,  # Sea State 9 + sensor platform (critical)
        'efficiency': 0.35,  # 15,000 NM range (high priority)
        'speed': 0.25,  # 28-30 kts capability (moderate priority)
    })

    # Pareto frontier objectives (multi-objective optimization)
    pareto_objectives: List[str] = field(default_factory=lambda: [
        'maximize_stability',  # GM, Sea State 9 performance
        'maximize_range',  # Fuel efficiency, endurance
        'maximize_payload',  # Mission module capacity
        'maximize_speed',  # Sprint capability
        'minimize_rcs',  # Radar cross-section
    ])

    # === DESIGN CONSTRAINTS ===
    hard_constraints: Dict[str, Any] = field(default_factory=lambda: {
        'loa': (46.0, 50.0),  # Meters (M48 hull proven)
        'beam_per_hull': (1.8, 2.2),  # Meters
        'hull_spacing': (8.0, 12.0),  # Meters
        'displacement': (90.0, 250.0),  # Metric tons
        'draft': (1.0, 2.5),  # Meters
        'design_speed': (15.0, 32.0),  # Knots
        'structural_weight': 83.5,  # Metric tons (92 short tons fixed)
    })

    # Navy certification requirements
    certification_standards: List[str] = field(default_factory=lambda: [
        'COLREGS_COMPLIANCE',  # Collision avoidance
        'UMAA_6',  # Unmanned Maritime Autonomy Architecture v6
        'NAVSEA_9310',  # Quality assurance
        'MIL_STD_1399',  # Electric power
        'MIL_STD_2042',  # Electromagnetic compatibility
    ])

    # === RESEARCH QUESTIONS ===
    # Questions the autonomous system will explore
    research_questions: List[str] = field(default_factory=lambda: [
        # Stability questions
        "What hull spacing optimizes stability for 4×36-ton containerized payload in Sea State 9?",
        "How does sensor mast height affect metacentric height (GM) and roll period?",
        "What deadrise angle balances sea-keeping vs. cargo deck volume?",

        # Range/efficiency questions
        "What is the Pareto frontier for fuel capacity vs. payload capacity?",
        "How does hull form (Cp, Cb) affect range at 15-20 knot cruise speed?",
        "What speed/payload combinations achieve >15,000 NM range?",

        # RCS reduction questions
        "What faceted geometry angles minimize radar cross-section without excessive drag?",
        "How does deadrise angle correlate with RCS reduction?",
        "What is the performance penalty for low-observable hull forms?",

        # Mission-specific questions
        "Optimal LCB position for heavy forward ISR sensors (MUOS antenna)?",
        "How does single-engine-out affect speed/range with twin diesel configuration?",
        "What deck volume/displacement ratios support UUV launch/recovery?",

        # Multi-objective questions
        "What design variants lie on the Pareto frontier (stability × range × payload)?",
        "How sensitive is performance to hull spacing in the 8-12m range?",
        "What configurations best replace Independence-class LCS mission modules?",
    ])

    # === SEA TRIAL CALIBRATION ===
    sea_trial_data_summary: Dict[str, Any] = field(default_factory=lambda: {
        'total_distance_nm': 32000,  # Nautical miles
        'ocean_regions': ['Pacific Ocean', 'Caribbean Sea', 'US East Coast'],
        'max_sea_state_proven': 9,  # WMO classification
        'max_wind_speed_kts': 68,  # Knots (Sea State 9)
        'max_wave_height_ft': 40,  # Feet (12m)
        'proven_speed_range': (15, 30),  # Knots
        'proven_range_nm': 15000,  # At cruise speed
        'propulsion_hours': 2500,  # Hours at sea
        'autonomy_stack': 'L3Harris ASView (TRL-9)',
    })

    # === AGENT SYSTEM CONFIGURATION ===
    agent_focus_areas: Dict[str, List[str]] = field(default_factory=lambda: {
        'explorer': [
            'Pareto frontier discovery (stability × range × payload)',
            'Mission-specific variants (ISR, ASW/MCM, picket)',
            'Sensitivity analysis (hull spacing, Cp, Cb)',
        ],
        'architect': [
            'NAVSEA-compliant designs (containerized payload)',
            'RCS-reduced hull forms (faceted geometry)',
            'Multi-mission adaptability (modular payload integration)',
        ],
        'critic': [
            'Sea State 9 stability validation',
            'Navy certification compliance (COLREGS, UMAA 6)',
            'Payload safety margins (144-ton container loads)',
        ],
        'historian': [
            'Sea trial data integration (32,000 NM calibration)',
            'Proven performance envelope documentation',
            'Empirical resistance corrections (ITTC-1957 adjustments)',
        ],
        'supervisor': [
            'LCS mission replacement assessment',
            'Fleet integration strategy (distributed operations)',
            'Program risk mitigation (proven vs. novel designs)',
        ],
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'program_name': self.program_name,
            'platform_name': self.platform_name,
            'solicitation_number': self.solicitation_number,
            'vessel_class': self.vessel_class,
            'primary_missions': self.primary_missions,
            'secondary_missions': self.secondary_missions,
            'threshold_requirements': self.threshold_requirements,
            'objective_requirements': self.objective_requirements,
            'optimization_weights': self.optimization_weights,
            'pareto_objectives': self.pareto_objectives,
            'hard_constraints': self.hard_constraints,
            'certification_standards': self.certification_standards,
            'research_questions': self.research_questions,
            'sea_trial_data_summary': self.sea_trial_data_summary,
            'agent_focus_areas': self.agent_focus_areas,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'M48MissionConfig':
        """Create configuration from dictionary."""
        return cls(**data)

    def get_agent_context_block(self) -> str:
        """
        Generate context block for agent prompts.

        Returns a formatted string that can be injected into agent system prompts
        to provide M48 mission context.
        """
        return f"""
=== M48 MISSION CONTEXT ===

Platform: {self.platform_name}
Program: {self.program_name} ({self.solicitation_number})
Class: {self.vessel_class}

PRIMARY MISSIONS: {', '.join(self.primary_missions)}

PROVEN PERFORMANCE (32,000 NM Sea Trials):
- Range: {self.sea_trial_data_summary['proven_range_nm']:,} NM
- Speed: {self.sea_trial_data_summary['proven_speed_range'][0]}-{self.sea_trial_data_summary['proven_speed_range'][1]} knots
- Sea State: {self.sea_trial_data_summary['max_sea_state_proven']} (winds >{self.sea_trial_data_summary['max_wind_speed_kts']} kts)
- Ocean Regions: {', '.join(self.sea_trial_data_summary['ocean_regions'])}

THRESHOLD REQUIREMENTS:
- Payload: {self.threshold_requirements['payload_capacity']:.0f} metric tons (4×36-ton containers)
- Range: {self.threshold_requirements['min_range']:,} NM
- Speed: {self.threshold_requirements['max_speed_threshold']:.0f} knots
- Sea State: {self.threshold_requirements['sea_state']}

OPTIMIZATION OBJECTIVES:
- Stability: {self.optimization_weights['stability']:.0%} weight (Sea State 9 + sensor platforms)
- Efficiency: {self.optimization_weights['efficiency']:.0%} weight (15,000 NM range)
- Speed: {self.optimization_weights['speed']:.0%} weight (28-30 kts capability)

TARGET MISSION:
Replace Independence-class LCS mission modules with distributed mUSV fleet:
- Smaller platform (48m vs 127m LCS)
- Lower cost (1/10th acquisition cost)
- Higher availability (autonomous operations)
- Persistent presence (15,000 NM endurance)

KEY DESIGN TRADEOFFS:
1. Fuel vs. Payload (range vs. mission capability)
2. Stability vs. Speed (GM vs. slenderness)
3. RCS Reduction vs. Drag (faceted geometry cost)
4. Deck Volume vs. Draft (containerized payload vs. port access)
"""

    def summary(self) -> str:
        """Generate human-readable summary of M48 mission configuration."""
        lines = [
            "=" * 70,
            "M48 MISSION CONFIGURATION",
            "=" * 70,
            "",
            f"Program: {self.program_name}",
            f"Platform: {self.platform_name}",
            f"Solicitation: {self.solicitation_number}",
            "",
            f"Primary Missions: {', '.join(self.primary_missions[:3])} (+{len(self.primary_missions)-3} more)",
            "",
            "OPTIMIZATION OBJECTIVES:",
            f"  Stability:   {self.optimization_weights['stability']:.0%} weight",
            f"  Efficiency:  {self.optimization_weights['efficiency']:.0%} weight",
            f"  Speed:       {self.optimization_weights['speed']:.0%} weight",
            "",
            "PARETO OBJECTIVES:",
        ]

        for obj in self.pareto_objectives:
            lines.append(f"  - {obj.replace('_', ' ').title()}")

        lines.extend([
            "",
            "SEA TRIAL VALIDATION:",
            f"  Total Distance: {self.sea_trial_data_summary['total_distance_nm']:,} NM",
            f"  Max Sea State: {self.sea_trial_data_summary['max_sea_state_proven']}",
            f"  Proven Range: {self.sea_trial_data_summary['proven_range_nm']:,} NM",
            f"  Ocean Regions: {len(self.sea_trial_data_summary['ocean_regions'])}",
            "",
            f"RESEARCH QUESTIONS: {len(self.research_questions)}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


def get_default_m48_config() -> M48MissionConfig:
    """Get default M48 mission configuration."""
    return M48MissionConfig()


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("M48 MISSION CONFIGURATION - DEMONSTRATION")
    print("=" * 70)
    print()

    # Create default configuration
    config = get_default_m48_config()

    # Show summary
    print(config.summary())
    print()

    # Show agent context block
    print("=" * 70)
    print("AGENT CONTEXT BLOCK (for injection into system prompts):")
    print("=" * 70)
    print(config.get_agent_context_block())

    # Show research questions
    print()
    print("=" * 70)
    print("RESEARCH QUESTIONS FOR AUTONOMOUS EXPLORATION:")
    print("=" * 70)
    print()

    for i, question in enumerate(config.research_questions, 1):
        print(f"{i}. {question}")

    print()
    print("=" * 70)
    print()

    # Export to JSON
    print("Exporting configuration to JSON...")
    json_str = config.to_json()
    print(f"JSON length: {len(json_str)} characters")
    print()

    # Test serialization/deserialization
    config_reloaded = M48MissionConfig.from_dict(json.loads(json_str))
    print(f"✓ Serialization successful: {config_reloaded.platform_name}")

    print()
    print("=" * 70)
