"""Keep six controller histories; replace raw ages by normalization means."""
from .reference_gru_controller_history import ReferenceGRUControllerHistory


class ReferenceGRUControllerNoAge(ReferenceGRUControllerHistory):
    def forward(self,history_controller,**kwargs):
        # (mean - mean) / scale is exactly zero after the inherited normalization.
        masked=history_controller.clone()
        masked[...,6:]=self.controller_mean[6:]
        return super().forward(history_controller=masked,**kwargs)
