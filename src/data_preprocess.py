import os
import numpy as np
from mido import MidiFile

# Simple mapping - tune to your MIDI drum mapping if needed
INSTR_MAP = {
    36: "piano",   # example mapping; adjust to real notes
    38: "repique",
    42: "chico",
}

def midi_to_grid(midi_path, resolution=16, bars=4):
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    steps = bars * resolution
    ticks_per_step = (ticks_per_beat * 4) / resolution
    events = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note = msg.note
                if note in INSTR_MAP:
                    events.append((abs_time, INSTR_MAP[note], msg.velocity))
    arr = np.zeros((steps, 3), dtype=np.uint8)  # 3 instruments
    idx_map = {"chico":0, "repique":1, "piano":2}
    for abs_ticks, instr, vel in events:
        step = int(abs_ticks / ticks_per_step)
        if 0 <= step < steps:
            arr[step, idx_map[instr]] = min(127, vel)
    return arr

def save_npz(arr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, data=arr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/data_preprocess.py in.mid out.npz")
    else:
        arr = midi_to_grid(sys.argv[1])
        save_npz(arr, sys.argv[2])
        print("Saved", sys.argv[2])