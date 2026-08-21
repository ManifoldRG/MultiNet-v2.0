"""Start splash and end-of-episode overlays for the human-play demo."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from demo.compare import TaskComparison, r1_task_id
from demo.theme import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    APP_SUBTITLE,
    APP_TITLE,
    COLOR_BG,
    COLOR_SEPARATOR,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_TEXT_LABEL,
    COLOR_TEXT_SUBTITLE,
    COLOR_TEXT_TITLE,
    GRID_DISPLAY_SIZE,
    LEFT_RAIL_W,
    MAX_DIFFICULTY_TIER,
    STATUS_MOVES_CRIT,
    TOP_BAR_H,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    difficulty_color,
)

if TYPE_CHECKING:
    from demo.ui import MiniGridPlayerUI


def _score_color(pct: int) -> tuple:
    if pct >= 100:
        return ACCENT_GREEN
    if pct >= 80:
        return ACCENT_CYAN
    if pct >= 50:
        return ACCENT_AMBER
    return STATUS_MOVES_CRIT


def _compose_colored_line(
    font: pygame.font.Font,
    parts: list[tuple[str, tuple]],
) -> pygame.Surface:
    """Render adjacent ``(text, color)`` segments on one horizontal line."""
    segments = [font.render(text, True, color) for text, color in parts if text]
    if not segments:
        return font.render("", True, COLOR_TEXT_SUBTITLE)
    width = sum(s.get_width() for s in segments)
    height = max(s.get_height() for s in segments)
    line = pygame.Surface((width, height), pygame.SRCALPHA)
    x = 0
    for segment in segments:
        line.blit(segment, (x, 0))
        x += segment.get_width()
    return line


def _wrap_text_surfs(
    text: str,
    font: pygame.font.Font,
    color: tuple,
    max_width: int,
    max_lines: int = 3,
) -> list[pygame.Surface]:
    words = text.split()
    lines: list[str] = []
    current = ""
    overflow = False
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            overflow = True
            current = ""
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    elif current:
        overflow = True
    if overflow and lines:
        ellipsized = lines[-1].rstrip(". ") + "..."
        while ellipsized and font.size(ellipsized)[0] > max_width:
            ellipsized = ellipsized[:-4] + "..."
        lines[-1] = ellipsized
    return [font.render(line, True, color) for line in lines]


def render_start_screen(ui: "MiniGridPlayerUI") -> None:
    ui.screen.fill(COLOR_BG)

    wash = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    for i in range(WINDOW_HEIGHT):
        t = i / max(1, WINDOW_HEIGHT - 1)
        alpha = int(28 * (1.0 - abs(t - 0.42) * 1.6))
        if alpha > 0:
            pygame.draw.line(wash, (70, 78, 110, alpha), (0, i), (WINDOW_WIDTH, i))
    ui.screen.blit(wash, (0, 0))

    cx = WINDOW_WIDTH // 2
    cy = WINDOW_HEIGHT // 2

    title = ui.font_splash_title.render(APP_TITLE, True, COLOR_TEXT_TITLE)
    ui.screen.blit(title, title.get_rect(center=(cx, cy - 48)))

    subtitle = ui.font_splash_sub.render(APP_SUBTITLE, True, COLOR_TEXT_SUBTITLE)
    ui.screen.blit(subtitle, subtitle.get_rect(center=(cx, cy + 8)))

    pulse = (math.sin(pygame.time.get_ticks() / 420.0) + 1.0) * 0.5
    prompt_color = tuple(
        int(COLOR_TEXT_DIM[i] + (COLOR_TEXT[i] - COLOR_TEXT_DIM[i]) * pulse)
        for i in range(3)
    )
    prompt = ui.font_splash_prompt.render("Press Enter to play", True, prompt_color)
    ui.screen.blit(prompt, prompt.get_rect(center=(cx, cy + 78)))

    hint = ui.font_small_bold.render("Q to quit", True, COLOR_TEXT_DIM)
    ui.screen.blit(hint, hint.get_rect(center=(cx, WINDOW_HEIGHT - 36)))


def render_episode_overlay(ui: "MiniGridPlayerUI") -> None:
    session = ui.session
    if not session.episode_done:
        return
    if session.episode_success and not ui.fx.success_overlay_ready():
        return

    task_id = r1_task_id(session.task_path)
    if task_id not in ui.r1_catalog:
        # No results table at all (e.g. a fresh clone with no
        # Multinet-v2-results checkout), or this task simply isn't an R1
        # maze -- either way there's no TaskComparison to draw the
        # model-comparison rows from. Never raise out of the render path:
        # show a plain end card instead.
        render_plain_result_overlay(ui)
        return
    comparison = ui.r1_catalog.lookup(task_id)
    render_r1_result_overlay(ui, comparison)


def render_plain_result_overlay(ui: "MiniGridPlayerUI") -> None:
    """Non-comparison end card: steps + success/failure only.

    Used whenever ``render_episode_overlay`` finds no R1 catalog entry for
    the finished task. Mirrors ``render_r1_result_overlay``'s header /
    description / difficulty / steps-line layout (computed locally from the
    session, same as that function); omits the BFS-optimal/score-vs-models
    rows since there is no ``TaskComparison`` to draw them from.
    """
    session = ui.session
    state = session.state
    human_steps = state.step_count
    max_steps = state.max_steps
    success = session.episode_success
    spec = session.task_spec
    desc = (spec.description or "").strip() if spec else ""
    tier = spec.difficulty_tier if spec else 0

    veil = pygame.Surface((GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE), pygame.SRCALPHA)
    veil.fill((8, 9, 13, 210))
    ui.screen.blit(veil, (LEFT_RAIL_W, TOP_BAR_H))

    pad = 28
    card_w = GRID_DISPLAY_SIZE - 2 * pad
    card_h = GRID_DISPLAY_SIZE - 2 * pad
    card_x = LEFT_RAIL_W + pad
    card_y = TOP_BAR_H + pad
    card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card, (22, 24, 34, 250), card.get_rect(), border_radius=16)
    accent = ACCENT_GREEN if success else (255, 110, 110)
    pygame.draw.rect(card, accent, pygame.Rect(0, 0, 7, card_h), border_radius=3)

    cx = card_w // 2
    content_w = card_w - 56
    items: list[tuple[pygame.Surface | None, int]] = []
    gap = 18

    headline = "YOU SOLVED IT" if success else (
        "STALLED" if session.end_reason == "stalled" else "OUT OF STEPS"
    )
    items.append((ui.font_overlay.render(headline, True, accent), gap + 6))

    if desc:
        desc_surfs = _wrap_text_surfs(desc, ui.font_small, COLOR_TEXT, content_w, max_lines=4)
        for i, surf in enumerate(desc_surfs):
            after = gap if i == len(desc_surfs) - 1 else 8
            items.append((surf, after))

    if tier:
        diff = f"Maze difficulty: {tier} / {MAX_DIFFICULTY_TIER}"
        items.append((ui.font_small_bold.render(diff, True, difficulty_color(tier)), gap + 4))

    if success:
        line = f"Completed in {human_steps} steps"
    else:
        line = f"Used {human_steps} / {max_steps} steps"
    items.append((ui.font_main_bold.render(line, True, COLOR_TEXT_TITLE), gap - 2))

    score_pct = int(round(session.display_reward * 100))
    score_line = _compose_colored_line(
        ui.font_main_bold,
        [(f"Score: {score_pct}%", _score_color(score_pct))],
    )
    items.append((score_line, gap + 4))

    items.append((
        ui.font_small_bold.render("R reset   [ ] switch task   Q quit", True, COLOR_TEXT_DIM),
        0,
    ))

    total_h = sum((2 if surf is None else surf.get_height()) + after for surf, after in items)
    y = max(24, (card_h - total_h) // 2)

    for surf, after in items:
        if surf is None:
            pygame.draw.line(
                card, COLOR_SEPARATOR,
                (cx - content_w // 2, y + 1), (cx + content_w // 2, y + 1),
            )
            y += 2 + after
        else:
            card.blit(surf, surf.get_rect(midtop=(cx, y)))
            y += surf.get_height() + after

    ui.screen.blit(card, (card_x, card_y))


def render_r1_result_overlay(ui: "MiniGridPlayerUI", comparison: TaskComparison) -> None:
    session = ui.session
    state = session.state
    human_steps = state.step_count
    max_steps = state.max_steps
    success = session.episode_success
    optimal = comparison.optimal_steps
    spec = session.task_spec
    desc = (spec.description or "").strip() if spec else ""
    tier = spec.difficulty_tier if spec else 0

    veil = pygame.Surface((GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE), pygame.SRCALPHA)
    veil.fill((8, 9, 13, 210))
    ui.screen.blit(veil, (LEFT_RAIL_W, TOP_BAR_H))

    pad = 28
    card_w = GRID_DISPLAY_SIZE - 2 * pad
    card_h = GRID_DISPLAY_SIZE - 2 * pad
    card_x = LEFT_RAIL_W + pad
    card_y = TOP_BAR_H + pad
    card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(card, (22, 24, 34, 250), card.get_rect(), border_radius=16)
    accent = ACCENT_GREEN if success else (255, 110, 110)
    pygame.draw.rect(card, accent, pygame.Rect(0, 0, 7, card_h), border_radius=3)

    cx = card_w // 2
    content_w = card_w - 56
    items: list[tuple[pygame.Surface | None, int]] = []
    gap = 18

    headline = "YOU SOLVED IT" if success else (
        "STALLED" if session.end_reason == "stalled" else "OUT OF STEPS"
    )
    items.append((ui.font_overlay.render(headline, True, accent), gap + 6))

    if desc:
        desc_surfs = _wrap_text_surfs(desc, ui.font_small, COLOR_TEXT, content_w, max_lines=4)
        for i, surf in enumerate(desc_surfs):
            after = gap if i == len(desc_surfs) - 1 else 8
            items.append((surf, after))

    if tier:
        diff = f"Maze difficulty: {tier} / {MAX_DIFFICULTY_TIER}"
        items.append((ui.font_small_bold.render(diff, True, difficulty_color(tier)), gap + 4))

    if success:
        line = f"Completed in {human_steps} steps"
    else:
        line = f"Used {human_steps} / {max_steps} steps"
    items.append((ui.font_main_bold.render(line, True, COLOR_TEXT_TITLE), gap - 2))

    if success and human_steps <= optimal:
        opt_detail = f"Optimal (BFS): {optimal} steps  -  you matched it"
    elif success:
        over = human_steps - optimal
        opt_detail = f"Optimal (BFS): {optimal} steps  -  you were {over} over"
    else:
        opt_detail = f"Optimal (BFS): {optimal} steps"

    score_pct = int(round(session.display_reward * 100))
    score_text = f"  -  Score: {score_pct}%"
    opt_score_line = _compose_colored_line(
        ui.font_main_bold,
        [
            (opt_detail, COLOR_TEXT_SUBTITLE),
            (score_text, _score_color(score_pct)),
        ],
    )
    items.append((opt_score_line, gap + 4))

    beat_count = sum(
        1 for m in comparison.models
        if success and (not m.success or human_steps < m.steps)
    )
    if success and beat_count == len(comparison.models):
        frame, frame_color = "You beat every model on this maze.", ACCENT_GREEN
    elif success and any(m.success for m in comparison.models):
        frame, frame_color = "How you stacked up against the models", COLOR_TEXT_SUBTITLE
    elif success:
        frame, frame_color = "None of the models solved this one.", ACCENT_GREEN
    else:
        frame, frame_color = "How the models did on this maze", COLOR_TEXT_SUBTITLE
    items.append((ui.font_main_bold.render(frame, True, frame_color), gap))
    items.append((None, gap))

    name_col_w = 100
    table_w = min(content_w, 460)
    row_h = 32
    rows: list[tuple[str, str, tuple]] = []
    you_color = ACCENT_GREEN if success else (255, 140, 140)
    you_detail = f"{human_steps} steps  -  solved" if success else f"{human_steps} steps  -  failed"
    rows.append(("You", you_detail, you_color))
    for model in comparison.models:
        if model.success:
            result_color = ACCENT_GREEN
            if success and human_steps < model.steps:
                detail = f"{model.steps} steps  -  you were faster"
            elif success and human_steps == model.steps:
                detail = f"{model.steps} steps  -  tied"
            else:
                detail = f"{model.steps} steps  -  solved"
        else:
            result_color = (255, 140, 140)
            detail = model.summary_line
        rows.append((model.display_name, detail, result_color))

    table_h = 22 + row_h * len(rows)
    table = pygame.Surface((table_w, table_h), pygame.SRCALPHA)
    table.blit(ui.font_label.render("P L A Y E R", True, COLOR_TEXT_LABEL), (0, 0))
    table.blit(ui.font_label.render("R E S U L T", True, COLOR_TEXT_LABEL), (name_col_w, 0))
    ty = 22
    detail_max = table_w - name_col_w
    for name, detail, color in rows:
        name_color = color if name == "You" else COLOR_TEXT
        table.blit(ui.font_main_bold.render(name, True, name_color), (0, ty))
        table.blit(ui._fit_text(detail, ui.font_small_bold, color, detail_max), (name_col_w, ty + 2))
        ty += row_h
    items.append((table, gap + 8))
    items.append((
        ui.font_small_bold.render("R reset   [ ] switch task   Q quit", True, COLOR_TEXT_DIM),
        0,
    ))

    total_h = sum((2 if surf is None else surf.get_height()) + after for surf, after in items)
    y = max(24, (card_h - total_h) // 2)

    for surf, after in items:
        if surf is None:
            pygame.draw.line(
                card, COLOR_SEPARATOR,
                (cx - content_w // 2, y + 1), (cx + content_w // 2, y + 1),
            )
            y += 2 + after
        else:
            card.blit(surf, surf.get_rect(midtop=(cx, y)))
            y += surf.get_height() + after

    ui.screen.blit(card, (card_x, card_y))
