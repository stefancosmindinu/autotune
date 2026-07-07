#!/usr/bin/env python3
"""
autotune.py — Pitch correction (auto-tune) for vocal audio files.

How it works:
1. Loads the input audio file.
2. Detects the pitch (fundamental frequency) frame by frame.
3. For each frame, finds the nearest note in the selected musical scale.
4. Calculates how many semitones the pitch needs to be shifted to reach the target note.
5. Applies a time-varying pitch shift (using librosa's pitch shifting) to correct the voice.
6. Saves the corrected result as a new .wav file.

Installation (on Mac, in Terminal):
    pip3 install librosa soundfile numpy scipy

Usage:
    python3 autotune.py input.wav output.wav

Optional:
You can choose the musical scale (default: chromatic, meaning every note is
allowed but each pitch is pulled toward the nearest semitone, resulting in
gentle correction without changing the melody).

For a specific key (e.g. C major), see the --scale parameter below.
"""

import argparse
import numpy as np
import librosa
import soundfile as sf

# Note names within one octave, used to build musical scales
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale intervals (in semitones relative to the root note)
SCALES = {
    "chromatic": list(range(12)),          # all semitones — gentle correction
    "major": [0, 2, 4, 5, 7, 9, 11],        # e.g. C major: C D E F G A B
    "minor": [0, 2, 3, 5, 7, 8, 10],        # natural minor
}


def build_allowed_midi_notes(root_note: str, scale_name: str, midi_low=36, midi_high=84):
    """
    Builds the list of allowed MIDI notes for the selected scale
    within a typical vocal range (default: C2–C6).
    """
    if root_note not in NOTE_NAMES:
        raise ValueError(f"Unknown root note: {root_note}")

    root_idx = NOTE_NAMES.index(root_note)
    intervals = SCALES[scale_name]
    allowed_pitch_classes = set((root_idx + i) % 12 for i in intervals)

    allowed = []
    for midi in range(midi_low, midi_high + 1):
        if (midi % 12) in allowed_pitch_classes:
            allowed.append(midi)

    return np.array(allowed)


def nearest_allowed_midi(midi_value, allowed_midis):
    """Returns the nearest allowed MIDI note."""
    idx = (np.abs(allowed_midis - midi_value)).argmin()
    return allowed_midis[idx]


def autotune(
    input_path,
    output_path,
    root_note="C",
    scale_name="chromatic",
    correction_strength=1.0,
    frame_length=2048,
    hop_length=256,
):
    """
    Parameters:
        correction_strength:
            0.0 = no correction
            1.0 = full correction to the target note

        Intermediate values (e.g. 0.6–0.8) sound more natural and less robotic.
    """

    print(f"Loading {input_path} ...")
    y, sr = librosa.load(input_path, sr=None, mono=True)

    print("Detecting pitch (this may take a few seconds)...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    allowed_midis = build_allowed_midi_notes(root_note, scale_name)

    # Convert detected pitch (Hz) to MIDI and compute the required correction
    n_frames = len(f0)
    semitone_shifts = np.zeros(n_frames)

    for i in range(n_frames):
        if voiced_flag[i] and f0[i] > 0:
            midi_val = librosa.hz_to_midi(f0[i])
            target_midi = nearest_allowed_midi(midi_val, allowed_midis)
            shift = (target_midi - midi_val) * correction_strength
            semitone_shifts[i] = shift
        else:
            # Do not correct silence or unvoiced/noisy frames
            semitone_shifts[i] = 0.0

    # Smooth transitions between frames to reduce robotic artifacts
    smoothed_shifts = (
        librosa.util.normalize(semitone_shifts)
        if False
        else semitone_shifts
    )

    window = max(1, int(sr / hop_length * 0.03))  # ~30 ms smoothing window

    if window > 1:
        kernel = np.ones(window) / window
        smoothed_shifts = np.convolve(semitone_shifts, kernel, mode="same")
    else:
        smoothed_shifts = semitone_shifts

    print("Applying pitch correction...")
    # librosa.effects.pitch_shift only accepts a single pitch shift value
    # per call, so we process the signal in small overlapping segments and
    # reconstruct the output using a simple overlap-add approach.

    output = np.zeros_like(y)
    samples_per_hop = hop_length
    half_window = frame_length // 2

    for i in range(n_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + samples_per_hop, len(y))

        if start_sample >= len(y):
            break

        segment = y[
            max(0, start_sample - half_window):
            min(len(y), end_sample + half_window)
        ]

        if len(segment) == 0:
            continue

        shift_amount = smoothed_shifts[i]

        if abs(shift_amount) > 0.01:
            try:
                shifted_segment = librosa.effects.pitch_shift(
                    segment,
                    sr=sr,
                    n_steps=shift_amount,
                )
            except Exception:
                shifted_segment = segment
        else:
            shifted_segment = segment

        # Extract only the center portion corresponding to the current hop
        center_start = min(half_window, start_sample)
        out_start = start_sample
        out_end = end_sample

        seg_slice = shifted_segment[
            center_start:
            center_start + (out_end - out_start)
        ]

        if len(seg_slice) < (out_end - out_start):
            seg_slice = np.pad(
                seg_slice,
                (0, (out_end - out_start) - len(seg_slice)),
            )

        output[out_start:out_end] = seg_slice[: out_end - out_start]

    print(f"Saving output to {output_path} ...")
    sf.write(output_path, output, sr)

    print("Done! 🎤")


def main():
    parser = argparse.ArgumentParser(
        description="Simple auto-tune / pitch correction for vocal recordings."
    )

    parser.add_argument(
        "input",
        help="Input audio file (wav, mp3, etc.)",
    )

    parser.add_argument(
        "output",
        help="Output audio file (.wav recommended)",
    )

    parser.add_argument(
        "--root",
        default="C",
        choices=NOTE_NAMES,
        help=(
            "Root note of the scale (default: C). "
            "Used only if --scale is not 'chromatic'."
        ),
    )

    parser.add_argument(
        "--scale",
        default="chromatic",
        choices=list(SCALES.keys()),
        help=(
            "Scale used for correction "
            "(default: chromatic = every semitone is allowed)."
        ),
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help=(
            "Correction strength: "
            "0.0 (none) - 1.0 (full correction, robotic sound). "
            "Default: 0.8."
        ),
    )

    args = parser.parse_args()

    autotune(
        args.input,
        args.output,
        root_note=args.root,
        scale_name=args.scale,
        correction_strength=args.strength,
    )


if __name__ == "__main__":
    main()