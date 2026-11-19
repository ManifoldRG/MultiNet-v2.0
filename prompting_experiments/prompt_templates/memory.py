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
# Shown in place of user.CURRENT_IMAGE_PLACEHOLDER when the notes spell it: the
# runner expands that placeholder into the current image after the notes are in.
NOTES_IMAGE_PLACEHOLDER_STAND_IN = "(current_image)"
# After the section, at the step after the model's notes were cut to the cap.
SCRATCHPAD_TRUNCATED_NOTE = (
    "Your last notes had {chars} characters; only the first {max_chars} were kept."
)

# Replaces feedback.PARSE_FAILURE_FEEDBACK in the scratchpad arms, whose reply
# is the block and then the FINAL_OUTPUT line, not "exactly one line". Keeps
# its "Could not parse" opening (the runner places the feedback by it).
SCRATCHPAD_PARSE_FAILURE_FEEDBACK = (
    "Could not parse FINAL_OUTPUT. Do not explain. Reply exactly as your notes "
    "between <scratchpad> and </scratchpad>, then one line: "
    "FINAL_OUTPUT: <one of {actions_hint}>."
)

# Opens the reply format: inserted after OUTPUT_FORMAT_LEAD, the opening line
# of every querying mode's instruction (prompt_templates/querying.py).
OUTPUT_FORMAT_LEAD = "Output exactly:\n"
SCRATCHPAD_OUTPUT_FORMAT = "<scratchpad>\n<your notes>\n</scratchpad>\n"
