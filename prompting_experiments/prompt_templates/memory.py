"""Scratchpad memory templates (ExperimentConfig.memory; see interface/scratchpad.py).

Every word the scratchpad arms add to a prompt lives here. The full_context arm
adds none of it.
"""

SCRATCHPAD_OPEN = "<scratchpad>"
SCRATCHPAD_CLOSE = "</scratchpad>"

# One sentence after the Valid actions line of the system prompt.
SCRATCHPAD_SYSTEM_LINE = (
    "Scratchpad: write notes between <scratchpad> and </scratchpad> in each reply; "
    "they replace your previous notes and are shown to you at the next step "
    "(at most {max_chars} characters)."
)

# The current notes, in the current turn right before the question.
SCRATCHPAD_SECTION = "Your scratchpad:\n<scratchpad>\n{notes}</scratchpad>"
# After the section, at the step after the model's notes were cut to the cap.
SCRATCHPAD_TRUNCATED_NOTE = (
    "Your last notes had {chars} characters; only the first {max_chars} were kept."
)

# Opens the reply format after "Output exactly:", before the FINAL_OUTPUT line.
SCRATCHPAD_OUTPUT_FORMAT = "<scratchpad>\n<your notes>\n</scratchpad>\n"
