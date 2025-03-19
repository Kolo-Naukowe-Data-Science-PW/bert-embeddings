"""
This module provides utility functions and classes for processing MIDI files.

Classes:
    Item: Represents a musical item such as a note or tempo change.
    Event: Represents an event in the musical sequence.

Functions:
    read_items(file_path): Reads notes and tempo changes from a MIDI file.
    item2event(groups): Converts items to events.
    quantize_items(items, ticks): Quantizes the start and end times of items.
    group_items(items, max_time, ticks_per_bar): Groups items into bars.
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np
import miditoolkit

from src.constants import (DEFAULT_RESOLUTION, DEFAULT_FRACTION, DEFAULT_DURATION_BINS)

@dataclass
class Item:
    """Represents a musical item such as a note or tempo change."""
    name: str
    start: int
    end: Optional[int]
    velocity: Optional[int]
    pitch: int
    item_type: int

def read_items(file_path: str) -> (List[Item], List[Item]):
    """
    Reads notes and tempo changes from a MIDI file.

    Args:
        file_path: Path to the MIDI file.

    Returns:
        A tuple containing two lists: note items and tempo items.
    """
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)

    # Process notes
    note_items = []
    num_of_instr = len(midi_obj.instruments)

    for i in range(num_of_instr):
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            note_items.append(Item(
                name='Note',
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
                item_type=i))
    note_items.sort(key=lambda x: x.start)

    # Process tempo changes
    tempo_items: List[Item] = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo),
            item_type=-1))
    tempo_items.sort(key=lambda x: x.start)

    # Expand tempo to all beats
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick],
                item_type=-1))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch,
                item_type=-1))
    tempo_items = output
    return note_items, tempo_items

@dataclass
class Event:
    """Represents an event in the musical sequence."""
    name: str
    time: Optional[int]
    value: Any
    text: str
    event_type: int

def item2event(groups: List[List[Any]]) -> List[List[Event]]:
    """
    Converts groups of items to events.

    Args:
        groups: A list where each element is a group (list) of items including a start and end time.

    Returns:
        A list of lists of events.
    """
    events: List[List[Event]] = []
    downbeat_count = 0
    for group in groups:
        if 'Note' not in [item.name for item in group[1:-1]]:
            continue
        bar_start, bar_end = group[0], group[-1]
        downbeat_count += 1
        new_bar = True

        for item in group[1:-1]:
            if item.name != 'Note':
                continue
            note_events: List[Event] = []

            # Bar event
            bar_value = 'New' if new_bar else 'Continue'
            new_bar = False
            note_events.append(Event(
                name='Bar',
                time=None,
                value=bar_value,
                text=f'{downbeat_count}',
                event_type=-1))

            # Position event
            flags = np.linspace(bar_start, bar_end, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(np.abs(flags - item.start))
            note_events.append(Event(
                name='Position',
                time=item.start,
                value=f'{index + 1}/{DEFAULT_FRACTION}',
                text=f'{item.start}',
                event_type=-1))

            # Pitch event
            pitch_type = -1
            note_events.append(Event(
                name='Pitch',
                time=item.start,
                value=item.pitch,
                text=f'{item.pitch}',
                event_type=pitch_type))

            # Duration
            duration = item.end - item.start if item.end is not None else 0
            dur_index = int(np.argmin(np.abs(np.array(DEFAULT_DURATION_BINS) - duration)))
            note_events.append(Event(
                name='Duration',
                time=item.start,
                value=dur_index,
                text=f'{duration}/{DEFAULT_DURATION_BINS[dur_index]}',
                event_type=-1))

            events.append(note_events)
    return events

def quantize_items(items: List[Item], ticks: int = 120) -> List[Item]:
    """
    Quantizes the start and end times of items to the nearest grid defined by ticks.

    Args:
        items: List of items to quantize.
        ticks: The tick interval to quantize to.

    Returns:
        The quantized list of items.
    """

    # reconsider whether start + 1 is necessary
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        if item.end is not None:
            item.end += shift
    return items

def group_items(items: List[Item], max_time: int, ticks_per_bar: int = DEFAULT_RESOLUTION * 4)\
        -> List[List[Any]]:
    """
    Groups items into bars.

    Args:
        items: List of items to group.
        max_time: The maximum time tick.
        ticks_per_bar: Number of ticks per bar.

    Returns:
        A list of groups, where each group is a list with the bar start, the items within the bar,
        and the bar end.
    """
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = [item for item in items if db1 <= item.start < db2]
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups
