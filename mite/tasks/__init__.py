"""MITE task definitions."""

from mite.tasks.base import MITETask
from mite.tasks.entailment_interaction import SICKREntailmentTask
from mite.tasks.claim_verification import (
    FEVERInteractionTask,
    ClimateFEVERInteractionTask,
    SciFActInteractionTask,
)
from mite.tasks.answer_quality import FiQAInteractionTask, CQADupstackInteractionTask
from mite.tasks.advice_relevance import SummEvalInteractionTask

ALL_TASKS = [
    SICKREntailmentTask,
    FEVERInteractionTask,
    ClimateFEVERInteractionTask,
    SciFActInteractionTask,
    FiQAInteractionTask,
    CQADupstackInteractionTask,
    SummEvalInteractionTask,
]

__all__ = [
    "MITETask",
    "ALL_TASKS",
    "SICKREntailmentTask",
    "FEVERInteractionTask",
    "ClimateFEVERInteractionTask",
    "SciFActInteractionTask",
    "FiQAInteractionTask",
    "CQADupstackInteractionTask",
    "SummEvalInteractionTask",
]
