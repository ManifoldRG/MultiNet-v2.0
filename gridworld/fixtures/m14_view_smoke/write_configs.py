"""Write the 12 M14 2D/3D smoke configs. One maze, one model, one view each."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
MAZES = {
    "mix_all": "ogbench/ogbench/procgen/maze_jsons/M14/25x25_dense_kr_sg_kb_mix_all_0.json",
    "mix_commitment": "ogbench/ogbench/procgen/maze_jsons/M14/25x25_dense_kr_sg_kb_mix_commitment_0.json",
    "mix_recoverable": "ogbench/ogbench/procgen/maze_jsons/M14/25x25_dense_kr_sg_kb_mix_recoverable_0.json",
}
EXPERIMENT = {
    "prompting": "minimal",
    "observation": "image_only",
    "context_window": "full",
    "chat_history": "stateless",
    "in_context_learning": "zero_shot",
    "action_space": "egocentric",
    "querying": "step_by_step",
    "progress_stall_k": 30,
    "prompt_cache": True,
}
RENDER_3D = {"backend": "mujoco3d", "camera": "first_person", "start_map": True}

for maze, task in MAZES.items():
    for view in ("2d", "3d"):
        for model in ("astra", "opus"):
            if model == "astra":
                entry = {
                    "provider": "openai",
                    "model": "gpt-6-astra",
                    "max_tokens": 64000,
                    "reasoning_effort": "xhigh",
                    "service_tier": "flex",
                    "image_detail": "high",
                    "timeout": 1800,
                    "max_attempts": 30,
                    "spend_cap_usd": 80,
                    "tasks": [task],
                }
                key = "gpt6_astra"
            else:
                entry = {
                    "provider": "claude",
                    "model": "claude-opus-5-5",
                    "max_tokens": 64000,
                    "timeout": 1800,
                    "max_attempts": 8,
                    "enable_thinking": True,
                    "effort": "xhigh",
                    "tasks": [task],
                }
                key = "claude_opus55"
            config = {"experiment_config": EXPERIMENT, "models": {key: entry}}
            if view == "3d":
                config["render"] = RENDER_3D
            path = OUT / f"run_config.{maze}_{view}_{model}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(path.name)
