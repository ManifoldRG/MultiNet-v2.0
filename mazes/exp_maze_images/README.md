# Experimental Maze Images

This directory contains example visualizations of the experimental maze sets. Each folder represents a different configuration or scenario type.

## Folder Overview

### M1 - Multi-Agent/Mechanism Maze (Key Required)
- **Example**: `10x10_corridor_kr_0.png`
- **Characteristics**: Contains key-required mechanics (kr variant)
- **Sizes**: 8×8, 10×10, 14×14
- **Layouts**: Corridor-like and dense arrangements
- **Purpose**: Tests agent navigation with interactive mechanisms (keys, doors, switches, etc.)

### S1 - Empty Room Scenario
- **Example**: `8x8_empty_room_0.png`
- **Characteristics**: No walls, simple open space
- **Purpose**: Baseline test for agent movement in unrestricted environment

### S2 - Simple Corridor (8×8)
- **Example**: `8x8_corridor_0.png`
- **Characteristics**: Straight corridors with walls
- **Size**: 8×8
- **Purpose**: Tests navigation in simple linear layouts

### S3 - Medium Corridor (10×10)
- **Example**: `10x10_corridor_0.png`
- **Characteristics**: More complex corridor layout with more variation
- **Size**: 10×10
- **Purpose**: Tests navigation in moderately complex layouts

### S4 - Dense Medium Maze (10×10)
- **Example**: `10x10_dense_0.png`
- **Characteristics**: High wall density, many branching paths
- **Size**: 10×10
- **Purpose**: Tests pathfinding and decision-making in complex space

### S5 - Large Corridor (14×14)
- **Example**: `14x14_corridor_0.png`
- **Characteristics**: Corridor-style layout at larger scale
- **Size**: 14×14
- **Purpose**: Tests navigation in larger but structured environments

### S6 - Dense Large Maze (14×14)
- **Example**: `14x14_dense_0.png`
- **Characteristics**: High wall density in large space, most complex
- **Size**: 14×14
- **Purpose**: Tests pathfinding under maximum complexity

## Key Differences

| Category | M1 | S1-S6 |
|----------|----|----|
| **Mechanics** | Contains keys, doors, switches, gates | Simple movement + goal |
| **Interaction** | Requires mechanism solving | Direct pathfinding |
| **Complexity** | Variable (kr variants) | Structured progression |
| **Sizes** | Multi-scale (8-14×14) | Per-scenario (S1: 8×8, S2-S3: sizes vary, S4-S6: increase) |

## Usage

These example images correspond to JSON specifications in `../exp_maze_jsons/`. Each PNG is a rendered visualization of the corresponding maze layout.
