# Copyright (c) 2024, Peristaltic Conveyor Configuration Utility
# Configuration management and parameter tuning for peristaltic conveyor

import json
import numpy as np
from typing import Dict, Any, Optional


class ConveyorConfig:
    """Configuration manager for peristaltic conveyor demo"""
    
    # Default configurations for different scenarios
    DEFAULT_CONFIGS = {
        'basic': {
            'actuator_grid_size': 15,
            'membrane_size': 0.8,
            'actuator_height': 0.04,
            'actuator_radius': 0.015,
            'membrane_height': 0.12,
            'wave_amplitude': 0.025,
            'wave_frequency': 0.4,
            'wave_speed': 0.15,
            'wave_length': 0.3,
            'use_cloth': False,
            'physics_dt': 1.0/60.0,
            'render_dt': 1.0/60.0
        },
        
        'performance': {
            'actuator_grid_size': 20,
            'membrane_size': 1.0,
            'actuator_height': 0.05,
            'actuator_radius': 0.02,
            'membrane_height': 0.15,
            'wave_amplitude': 0.04,
            'wave_frequency': 0.8,
            'wave_speed': 0.25,
            'wave_length': 0.4,
            'use_cloth': True,
            'physics_dt': 1.0/120.0,
            'render_dt': 1.0/60.0
        },
        
        'research': {
            'actuator_grid_size': 25,
            'membrane_size': 1.2,
            'actuator_height': 0.06,
            'actuator_radius': 0.025,
            'membrane_height': 0.18,
            'wave_amplitude': 0.05,
            'wave_frequency': 1.0,
            'wave_speed': 0.3,
            'wave_length': 0.5,
            'use_cloth': True,
            'physics_dt': 1.0/240.0,
            'render_dt': 1.0/60.0
        },
        
        'rl_training': {
            'actuator_grid_size': 16,
            'membrane_size': 0.8,
            'actuator_height': 0.04,
            'actuator_radius': 0.02,
            'membrane_height': 0.12,
            'wave_amplitude': 0.03,
            'wave_frequency': 0.6,
            'wave_speed': 0.2,
            'wave_length': 0.35,
            'use_cloth': True,
            'physics_dt': 1.0/60.0,
            'render_dt': 1.0/30.0,  # Lower render rate for RL training
            'headless': True,
            'max_episode_length': 1000,
            'reward_shaping': {
                'displacement_weight': 1.0,
                'efficiency_weight': 0.5,
                'energy_penalty': 0.1,
                'height_penalty': 10.0
            }
        }
    }
    
    def __init__(self, config_name: str = 'performance'):
        """Initialize with a predefined configuration"""
        if config_name not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Unknown configuration: {config_name}. Available: {list(self.DEFAULT_CONFIGS.keys())}")
        
        self.config = self.DEFAULT_CONFIGS[config_name].copy()
        self.config_name = config_name
        
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
        
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration parameters"""
        self.config.update(updates)
        
    def set_parameter(self, key: str, value: Any):
        """Set a specific parameter"""
        self.config[key] = value
        
    def get_parameter(self, key: str, default: Any = None):
        """Get a specific parameter"""
        return self.config.get(key, default)
        
    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        config_data = {
            'config_name': self.config_name,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
            
    def load_config(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
            
        self.config_name = config_data.get('config_name', 'custom')
        self.config = config_data['config']
        
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        required_params = [
            'actuator_grid_size', 'membrane_size', 'actuator_height',
            'actuator_radius', 'membrane_height', 'wave_amplitude',
            'wave_frequency', 'wave_speed', 'physics_dt', 'render_dt'
        ]
        
        for param in required_params:
            if param not in self.config:
                print(f"Missing required parameter: {param}")
                return False
                
        # Validate parameter ranges
        validations = [
            ('actuator_grid_size', lambda x: 5 <= x <= 50),
            ('membrane_size', lambda x: 0.1 <= x <= 5.0),
            ('actuator_height', lambda x: 0.01 <= x <= 0.2),
            ('actuator_radius', lambda x: 0.005 <= x <= 0.1),
            ('membrane_height', lambda x: 0.05 <= x <= 1.0),
            ('wave_amplitude', lambda x: 0.005 <= x <= 0.2),
            ('wave_frequency', lambda x: 0.1 <= x <= 5.0),
            ('wave_speed', lambda x: 0.05 <= x <= 2.0),
            ('physics_dt', lambda x: 1.0/480.0 <= x <= 1.0/30.0),
            ('render_dt', lambda x: 1.0/120.0 <= x <= 1.0/15.0)
        ]
        
        for param, validator in validations:
            if not validator(self.config[param]):
                print(f"Parameter {param} = {self.config[param]} is out of valid range")
                return False
                
        return True
        
    def optimize_for_hardware(self, gpu_memory_gb: float = 8.0, cpu_cores: int = 8):
        """Optimize configuration based on available hardware"""
        if gpu_memory_gb < 4.0:
            # Low-end GPU optimization
            self.config.update({
                'actuator_grid_size': min(self.config['actuator_grid_size'], 15),
                'physics_dt': max(self.config['physics_dt'], 1.0/60.0),
                'use_cloth': False
            })
            print("Optimized for low-end GPU")
            
        elif gpu_memory_gb >= 12.0:
            # High-end GPU optimization
            self.config.update({
                'physics_dt': min(self.config['physics_dt'], 1.0/240.0),
                'use_cloth': True
            })
            print("Optimized for high-end GPU")
            
        # CPU optimization
        if cpu_cores < 4:
            self.config['render_dt'] = max(self.config['render_dt'], 1.0/30.0)
            
    def create_wave_pattern_config(self, pattern_type: str) -> Dict[str, Any]:
        """Create wave pattern configuration for specific transport behavior"""
        patterns = {
            'gentle': {
                'wave_amplitude': 0.02,
                'wave_frequency': 0.3,
                'wave_speed': 0.1
            },
            'aggressive': {
                'wave_amplitude': 0.06,
                'wave_frequency': 1.2,
                'wave_speed': 0.4
            },
            'precise': {
                'wave_amplitude': 0.025,
                'wave_frequency': 0.8,
                'wave_speed': 0.15
            },
            'multi_directional': {
                'wave_amplitude': 0.035,
                'wave_frequency': 0.6,
                'wave_speed': 0.2,
                'pattern_sequence': [
                    [1, 0], [0.707, 0.707], [0, 1], 
                    [-0.707, 0.707], [-1, 0], [-0.707, -0.707],
                    [0, -1], [0.707, -0.707]
                ]
            }
        }
        
        if pattern_type not in patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
        return patterns[pattern_type]
        
    def print_config(self):
        """Pretty print current configuration"""
        print(f"\n=== Peristaltic Conveyor Configuration: {self.config_name} ===")
        print("Physical Parameters:")
        print(f"  Grid Size: {self.config['actuator_grid_size']}x{self.config['actuator_grid_size']}")
        print(f"  Membrane Size: {self.config['membrane_size']}m x {self.config['membrane_size']}m")
        print(f"  Actuator Height: {self.config['actuator_height']*100:.1f}cm")
        print(f"  Actuator Radius: {self.config['actuator_radius']*100:.1f}cm")
        print(f"  Membrane Height: {self.config['membrane_height']*100:.1f}cm")
        
        print("\nWave Parameters:")
        print(f"  Amplitude: {self.config['wave_amplitude']*100:.1f}cm")
        print(f"  Frequency: {self.config['wave_frequency']:.1f}Hz")
        print(f"  Speed: {self.config['wave_speed']:.2f}m/s")
        print(f"  Length: {self.config.get('wave_length', 'N/A')}")
        
        print("\nSimulation Parameters:")
        print(f"  Physics DT: {self.config['physics_dt']*1000:.1f}ms")
        print(f"  Render DT: {self.config['render_dt']*1000:.1f}ms")
        print(f"  Use Cloth: {self.config['use_cloth']}")
        print(f"  Total Actuators: {self.config['actuator_grid_size']**2}")
        print("=" * 50)


def create_experiment_configs():
    """Create a set of experimental configurations for research"""
    experiments = {}
    
    base_config = ConveyorConfig('performance').get_config()
    
    # Amplitude study
    for i, amp in enumerate([0.02, 0.03, 0.04, 0.05, 0.06]):
        config = base_config.copy()
        config['wave_amplitude'] = amp
        experiments[f'amplitude_study_{i+1}'] = config
        
    # Frequency study
    for i, freq in enumerate([0.4, 0.6, 0.8, 1.0, 1.2]):
        config = base_config.copy()
        config['wave_frequency'] = freq
        experiments[f'frequency_study_{i+1}'] = config
        
    # Grid resolution study
    for i, size in enumerate([12, 16, 20, 24, 28]):
        config = base_config.copy()
        config['actuator_grid_size'] = size
        experiments[f'resolution_study_{i+1}'] = config
        
    return experiments


if __name__ == "__main__":
    # Example usage
    print("Peristaltic Conveyor Configuration Utility")
    print("Available configurations:", list(ConveyorConfig.DEFAULT_CONFIGS.keys()))
    
    # Create and display different configurations
    for config_name in ['basic', 'performance', 'research']:
        config = ConveyorConfig(config_name)
        config.print_config()
        
    # Create experimental configurations
    experiments = create_experiment_configs()
    print(f"\nGenerated {len(experiments)} experimental configurations")
    
    # Save example configuration
    config = ConveyorConfig('performance')
    config.save_config('conveyor_config_example.json')
    print("Saved example configuration to 'conveyor_config_example.json'")