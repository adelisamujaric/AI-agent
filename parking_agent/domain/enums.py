"""
Domain sloj - Enum-i za parking enforcement
"""
from enum import Enum


class ViolationType(Enum):
    """Tipovi parking prekršaja"""
    NEPROPISNO_NA_INVALIDSKOM = "NepropisnoParkirano_naInvalidskomMjesto"
    NEPROPISNO_PREKO_LINIJE = "NepropisnoParkirano_prekoLinije"
    NEPROPISNO_VAN_OKVIRA = "NepropisnoParkirano_vanOkviraParkinga"
    PARKIRANJE_NA_REZERVACIJI = "Parkiranje_na_rezervisanom_mjestu"


class DetectionStatus(Enum):
    """Status detekcije"""
    QUEUED = "Queued"  # Čeka na obradu
    PROCESSING = "Processing"  # U obradi
    OK = "OK"  # Nema prekršaja
    NEEDS_ZOOM = "NeedsZoom"  # Potrebna zoom slika za tablicu
    READY_TO_CONFIRM = "ReadyToConfirm"  # Spremno za potvrdu
    CONFIRMED = "Confirmed"  # Potvrđeno (za učenje)
    REJECTED = "Rejected"  # Odbijeno (false positive)


class LearningStatus(Enum):
    """Status modela/učenja"""
    NOT_ENOUGH_DATA = "NotEnoughData"
    TRAINING = "Training"
    SUCCESS = "Success"
    NO_IMPROVEMENT = "NoImprovement"
    ERROR = "Error"