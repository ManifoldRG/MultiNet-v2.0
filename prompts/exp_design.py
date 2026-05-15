"""Experiment 3 prompt condition-set registry."""

from .condition_set_1_prompt import CONDITION_SET as CONDITION_SET_1
from .condition_set_2_observation_format import CONDITION_SET as CONDITION_SET_2
from .condition_set_3_context_window import CONDITION_SET as CONDITION_SET_3
from .condition_set_4_action_space import CONDITION_SET as CONDITION_SET_4
from .condition_set_5_querying_strategy import CONDITION_SET as CONDITION_SET_5
from .condition_set_6_in_context_learning import CONDITION_SET as CONDITION_SET_6

CONDITION_SETS = {
	CONDITION_SET_1["name"]: CONDITION_SET_1,
	CONDITION_SET_2["name"]: CONDITION_SET_2,
	CONDITION_SET_3["name"]: CONDITION_SET_3,
	CONDITION_SET_4["name"]: CONDITION_SET_4,
	CONDITION_SET_5["name"]: CONDITION_SET_5,
	CONDITION_SET_6["name"]: CONDITION_SET_6,
}
