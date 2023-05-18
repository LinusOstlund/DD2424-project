import muspy as mp
import re
import numpy as np
import pathlib
import os
import pickle as pkl


def transpose_nmat(track_data: dict) -> dict:
    """
    Transpose the nmat to 0-tonic, by subtracting the tonic from the third (pitch) column
    TODO I do believe this method mutates the track_data, inplace, so the return statment is not needed
    """
    # untransposed_nmat = np.array(track_data["nmat"]).copy()
    nmat = np.array(track_data["nmat"])
    tonic = track_data["tonic"]

    # here is the transposition
    nmat[:, 2] -= tonic

    # assert that none of the notes are negative
    assert np.all(nmat >= 0)

    """
    # assert that old_nmat and nmat are not the same
    # (since the dataset only has tonics 2, 8, 11, this should always be true)
    assert not np.all(untransposed_nmat == nmat)
    """

    track_data["nmat"] = nmat

    return track_data


def nmat2midi(track_data: dict, scale=8, transpose=2) -> mp.Music:
    """
    Convert all pieces to midi and then to abc.
    """
    music = mp.Music()  # Create new Music object for each piece
    track = mp.Track()  # Create new Track object for each piece

    # this should transpose the nmat to the correct key
    track_data = transpose_nmat(track_data)

    # append midi-meta message of which key the piece is in
    # major tonics are [0,11], minor tonics are [12,23]
    # TODO this is not working!
    # key = 0 if track_data["mode"] == 'M' else 12
    # music.meta.append(mp.KeySignature(key=key))
    music.append(
        mp.KeySignature(time=0, root=0, fifths=5 if track_data["mode"] == "M" else -3)
    )

    music.append(mp.TimeSignature(time=0, numerator=4, denominator=4))

    for note in track_data["nmat"]:
        note = mp.Note(
            time=scale * note[0],
            pitch=note[2] + transpose * 12,
            duration=int(scale * (max(note[1] - note[0], 0.5))),
        )
        track.notes.append(note)

    music.tracks.append(track)  # Append the track to the current Music object

    return music


def convert_data(data: dict, midi_path: str, abc_path: str, save=True) -> None:
    pieces = list(data.keys())

    for i, track_name in enumerate(pieces):
        # get data
        track_data = data[track_name]
        # make mp.Music object
        music = nmat2midi(track_data)

        # create the midi
        track_name = f"piece_{i}"
        midi_name = f"{midi_path}/{track_name}.mid"
        music.write_midi(midi_name)

        # make the midi an abc
        abc_name = f"{abc_path}/{track_name}.abc"
        os.system(f"midi2abc {midi_name} -o {abc_name}")


def clean_abc(abc):
    # Regular expression pattern
    expressions = [
        "~",
        "|",
        "-",
        "\|[\[]?1|:\|[\[]2",  # first and second endings
        '"\(?[A-G|a-g][b|\#]?[m|d|a|7]?(6|7|b9)?\)?(/[A-G|a-g][b|\#]?[m|7]?)?"',  # chords between ""
        # "\|Y:\s*(?<=[ABCDEFG])\s(?=[ABCDEFG])?"
        "/?(2|3|4|5|6|7|8|9|12|16)",  # duration
        "M:\d/\d",  # meter
        "L:1/(4|8|16)",  # base duration
        "K:[ABCDEFG](maj|min|m|dor)?",  # key
        ":?\|:?",  # vertical bar or repetition
        "\([\d]|",  # triplets like this (3
        "[\^\_\=]?[ABCDEFG][,]?",  # low octave notes
        "[\^\_\=]?[abcdefg][']?",  # high octave notes
        "z",  # pause
        "(>|<)",  # broken rhythm
        "\[|\]",  # square brackets
    ]
    pattern = "|".join(expressions)

    REGEX = re.compile(pattern)

    # remove empty chords
    abc = re.sub('" "', "", abc)
    # remove spaces
    abc = re.sub(" ", "", abc)

    # replace + with # for sharps
    abc = re.sub(r"c\+", "c#", abc)
    abc = re.sub(r"d\+", "d#", abc)
    abc = re.sub(r"f\+", "f#", abc)
    abc = re.sub(r"g\+", "g#", abc)

    # replace :: with :||:
    abc = re.sub(r":[\|]?:", ":||:", abc)
    # replace \
    abc = re.sub(r"\\", "", abc)

    # remove all lines that start with %
    abc = re.sub(r"(?m)^\%.*\n?", "", abc)
    # remove everything between !
    abc = re.sub(r"![^!]+!", "", abc)

    # remove all lines that start with X:,T:,S:
    abc = re.sub(r"(?m)^[XTSPYNQR]:?.*\n?", "", abc)
    # print(abc)

    abc = re.sub(REGEX, lambda y: y.group() + " ", abc)
    # print(abc)

    return abc


# the script
with open("data/dataset.pkl", "rb") as f:
    data = pkl.load(f)

samples = dict(list(data.items())[:10])

# noice
cwd = pathlib.Path.cwd()
midi_dir = cwd / "midi"
abc_dir = cwd / "abc"

# create abc_dir if it doesn't exist
os.makedirs(abc_dir, exist_ok=True)
os.makedirs(midi_dir, exist_ok=True)

os.system(f"rm {midi_dir}/*.mid")
os.system(f"rm {abc_dir}/*.abc")

# convert the samples to midi and then to abc
convert_data(samples, midi_dir, abc_dir)

with open(f"{abc_dir}/piece_0.abc", "r") as f:
    data = f.read()
    data = data.replace("%", "\n%")
    data = data.replace("\n\n", "\n")
    clean_abc(data)
