from typing import Union, List

from verl.utils.tracking import Tracking


class ReasonRLTracking(Tracking):
    """
    Extended tracking class that inherits all functionality from base Tracking
    but adds automatic source code logging to wandb.
    """
    
    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        # Call parent constructor
        super().__init__(project_name, experiment_name, default_backend, config)
        
        # Add source code logging if wandb is being used
        if 'wandb' in self.logger:
            # Always save source code to wandb
            try:
                import wandb
                wandb.run.log_code(".")
                print("✅ Source code saved to wandb")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save source code to wandb: {e}")
