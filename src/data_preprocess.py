import argparse
import numpy as np
from mido import MidiFile

from representation import N_INSTR, STEPS_PER_BAR, save_piece

# Default mapping - tune to your MIDI drum mapping with --note-map
INSTR_MAP = {
    36: "piano",
    38: "repique",
    42: "chico",
}
IDX_MAP = {"chico": 0, "repique": 1, "piano": 2}


def midi_to_grid(midi_path, note_map=None, resolution=STEPS_PER_BAR, bars=None):
    """Velocities (S, 3) and offsets as a fraction of a step (NaN where silent).

    Assumes 4/4 and that the file starts on a downbeat. Notes are snapped to the
    nearest 16th and the remainder is kept as microtiming.
    """
    note_map = note_map or INSTR_MAP
    mid = MidiFile(midi_path)
    ticks_per_step = (mid.ticks_per_beat * 4) / resolution
    events = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in note_map:
                events.append((abs_time, note_map[msg.note], msg.velocity))

    last_step = max((int(round(t / ticks_per_step)) for t, _, _ in events), default=0)
    total_bars = bars or (last_step // resolution + 1)
    steps = total_bars * resolution
    velocities = np.zeros((steps, N_INSTR), dtype=np.uint8)
    offsets = np.full((steps, N_INSTR), np.nan, dtype=np.float32)
    for abs_ticks, instr, vel in events:
        exact = abs_ticks / ticks_per_step
        step = int(round(exact))
        inst = IDX_MAP[instr]
        if 0 <= step < steps and vel > velocities[step, inst]:
            velocities[step, inst] = min(127, vel)
            offsets[step, inst] = exact - step
    return velocities, offsets


def parse_note_map(value):
    if not value:
        return None
    return {int(note): name for note, name in (part.split("=") for part in value.split(","))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a MIDI candombe performance into a training .npz")
    parser.add_argument("midi")
    parser.add_argument("out")
    parser.add_argument("--note-map", default="", help="e.g. 36=piano,38=repique,42=chico")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--tempo", type=float, default=None)
    parser.add_argument("--style", default=None)
    args = parser.parse_args()
    velocities, offsets = midi_to_grid(args.midi, parse_note_map(args.note_map), bars=args.bars)
    save_piece(args.out, velocities, offsets_frac=offsets, tempo=args.tempo, style=args.style)
    print("Saved", args.out)
