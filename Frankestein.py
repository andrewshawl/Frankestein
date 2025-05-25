#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contrapunto_pro_mega.py
────────────────────────
Generador de contrapunto avanzado a 3 voces (1ª a 4ª especie),
con Backtracking y GUI Streamlit.

© 2025  Andrew / ChatGPT Demo — MIT License
"""
from __future__ import annotations

import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from music21 import (
    instrument,
    key,
    meter,
    note,
    pitch,
    stream,
)
from rich.logging import RichHandler

##############################################################################
# 0. LOGGING + SOUNDFONT DETECTION
##############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)
log = logging.getLogger("counterpoint")

# Busca SoundFont en variable de entorno o rutas comunes
SOUNDFONT = os.getenv(
    "SF2",
    next(
        (
            p
            for p in (
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                "/opt/homebrew/share/soundfonts/FluidR3_GM.sf2",
            )
            if Path(p).exists()
        ),
        "",
    ),
)

##############################################################################
# 1. CONSTANTES Y DATACLASS
##############################################################################
VOICE_NAMES   = ("T", "A", "S")  # Tenor, Alto, Soprano
SPECIES_LABEL = {1: "1ª", 2: "2ª", 3: "3ª", 4: "4ª"}
MAX_LEAP      = 9  # semitonos → 6ª mayor

# Consonancias renacentistas (mod 12)
CONSONANT = {0, 3, 4, 5, 7, 8, 9}

def is_consonant(st_int: int) -> bool:
    return abs(st_int) % 12 in CONSONANT

@dataclass
class VoiceDef:
    name: str
    species: int
    midi_lo: int
    midi_hi: int

@dataclass
class State:
    cf: List[int]
    voices: Dict[str, List[int]]
    lcm_grid: int

##############################################################################
# 2. GENERACIÓN DE CANTUS FIRMUS
##############################################################################
def generate_cantus(
    length: int,
    k: key.Key,
    rng: Tuple[int, int],
    seed: int | None = None,
) -> List[int]:
    if seed is not None and seed != 0:
        random.seed(seed)

    scale = k.getPitches()
    tonic_midi = pitch.Pitch(k.tonic).midi
    line: List[int] = [tonic_midi]

    while len(line) < length - 1:
        cur = line[-1]
        cands = [
            p.midi
            for p in scale
            if rng[0] <= p.midi <= rng[1] and abs(p.midi - cur) <= MAX_LEAP
        ]
        random.shuffle(cands)

        chosen = None
        for c in cands:
            if len(line) >= 2:
                if (c - line[-1]) * (line[-1] - line[-2]) > 0:
                    continue
            chosen = c
            break

        if chosen is None:
            return generate_cantus(length, k, rng, seed)
        line.append(chosen)

    line.append(tonic_midi)

    peak = max(line)
    if line.count(peak) > 1:
        return generate_cantus(length, k, rng, seed)

    return line

##############################################################################
# 3. INTERVALOS Y CURSOR
##############################################################################
def intervals_semitones(n1: int, n2: int) -> int:
    return (n2 - n1) % 12

def is_perfect(st: int) -> bool:
    return st in (0, 7)

def parallel_or_direct(old_a: int, old_b: int, new_a: int, new_b: int) -> bool:
    old_int = intervals_semitones(old_a, old_b)
    new_int = intervals_semitones(new_a, new_b)
    dir_a = new_a - old_a
    dir_b = new_b - old_b

    if is_perfect(old_int) and (old_int == new_int):
        if dir_a * dir_b > 0:
            return True

    if not is_perfect(old_int) and is_perfect(new_int):
        if dir_a * dir_b > 0:
            if old_b > old_a:
                if abs(new_b - old_b) > 2:
                    return True
            else:
                if abs(new_a - old_a) > 2:
                    return True
    return False

@dataclass
class VoiceCursor:
    vdef: VoiceDef
    notes: List[int]

    def last_note(self, offset=1) -> int | None:
        if len(self.notes) < offset:
            return None
        return self.notes[-offset]

    def melodic_ok(self, midi_val: int) -> bool:
        if not (self.vdef.midi_lo <= midi_val <= self.vdef.midi_hi):
            return False
        prev = self.last_note()
        if prev is not None and abs(midi_val - prev) > MAX_LEAP:
            return False

        prev2 = self.last_note(2)
        if prev is not None and prev2 is not None:
            if abs(prev - prev2) >= 7:
                if (midi_val - prev) * (prev - prev2) >= 0:
                    return False
                if abs(midi_val - prev) > 2:
                    return False

        return True

##############################################################################
# 4. CHEQUEOS POR ESPECIE
##############################################################################
def check_1st_species(
    note_this: int, note_cf: int, sub_beat: int, total_subs: int,
    vcur: VoiceCursor, note_prev: int | None, note_cf_prev: int | None
) -> bool:
    st_int = note_this - note_cf
    return is_consonant(st_int)

def check_2nd_species(
    note_this: int, note_cf: int, sub_beat: int, total_subs: int,
    vcur: VoiceCursor, note_prev: int | None, note_cf_prev: int | None
) -> bool:
    st_int = note_this - note_cf
    if sub_beat == 0:
        return is_consonant(st_int)
    else:
        if is_consonant(st_int):
            return True
        else:
            if note_prev is None:
                return False
            if abs(note_this - note_prev) > 2:
                return False
        return True

def check_3rd_species(
    note_this: int, note_cf: int, sub_beat: int, total_subs: int,
    vcur: VoiceCursor, note_prev: int | None, note_cf_prev: int | None
) -> bool:
    st_int = note_this - note_cf
    if sub_beat in (0, 2):
        return is_consonant(st_int)
    else:
        if is_consonant(st_int):
            return True
        else:
            if note_prev is None:
                return False
            if abs(note_this - note_prev) > 2:
                return False
        return True

def check_4th_species(
    note_this: int, note_cf: int, sub_beat: int, total_subs: int,
    vcur: VoiceCursor, note_prev: int | None, note_cf_prev: int | None
) -> bool:
    st_int = note_this - note_cf
    if sub_beat == 0:
        return True
    else:
        return is_consonant(st_int)

def validate_species(
    note_this: int,
    note_cf: int,
    sub_beat: int,
    total_subs: int,
    vcur: VoiceCursor,
    note_prev: int | None,
    note_cf_prev: int | None
) -> bool:
    sp = vcur.vdef.species
    if sp == 1:
        return check_1st_species(note_this, note_cf, sub_beat, total_subs, vcur, note_prev, note_cf_prev)
    elif sp == 2:
        return check_2nd_species(note_this, note_cf, sub_beat, total_subs, vcur, note_prev, note_cf_prev)
    elif sp == 3:
        return check_3rd_species(note_this, note_cf, sub_beat, total_subs, vcur, note_prev, note_cf_prev)
    elif sp == 4:
        return check_4th_species(note_this, note_cf, sub_beat, total_subs, vcur, note_prev, note_cf_prev)
    return False

##############################################################################
# 5. BACKTRACKING "PRO"
##############################################################################
def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)

def generate_counterpoint_pro(
    cf: List[int],
    voices: Dict[str, VoiceDef],
    max_tries: int = 5,
    progress_cb=lambda x: None,
) -> Dict[str, List[int]]:
    sp_subdiv_map = {1:1, 2:2, 3:4, 4:2}
    subs_list = [sp_subdiv_map[v.species] for v in voices.values()]
    lcm_val = 1
    for s in subs_list:
        lcm_val = lcm(lcm_val, s)

    grid = []
    for b in range(len(cf)):
        for sub in range(lcm_val):
            grid.append((b, sub))

    cursors = {vn: VoiceCursor(vdef, []) for vn, vdef in voices.items()}
    order = ["T", "A", "S"]

    def dfs(idx: int) -> bool:
        if idx == len(grid):
            return True

        beat, sub = grid[idx]
        cf_pitch = cf[beat]

        for vn in order:
            vcur = cursors[vn]
            subdiv_required = sp_subdiv_map[vcur.vdef.species]
            block_size = lcm_val // subdiv_required

            if sub % block_size != 0:
                continue

            note_index = beat * subdiv_required + (sub // block_size)
            if len(vcur.notes) > note_index:
                continue

            base_range = range(vcur.vdef.midi_lo, vcur.vdef.midi_hi+1)
            prev_n = vcur.last_note()
            if prev_n is not None:
                base_list = sorted(base_range, key=lambda x: abs(x - prev_n))
            else:
                base_list = list(base_range)

            assigned_ok = False
            for cand in base_list:
                if not vcur.melodic_ok(cand):
                    continue

                sub_beat_local = note_index % subdiv_required
                note_cf_prev = cf[beat - 1] if beat > 0 else None
                n_prev = vcur.last_note()

                if not validate_species(cand, cf_pitch, sub_beat_local,
                                       subdiv_required, vcur, n_prev, note_cf_prev):
                    continue

                for oname in order:
                    if oname == vn:
                        continue
                    ocur = cursors[oname]
                    sp2 = sp_subdiv_map[ocur.vdef.species]
                    blk2 = lcm_val // sp2
                    idx2 = beat * sp2 + (sub // blk2)

                    if idx2 >= len(ocur.notes):
                        continue

                    other_note = ocur.notes[idx2]

                    if vn == "T" and oname == "A" and cand > other_note:
                        break
                    if vn == "T" and oname == "S" and cand > other_note:
                        break
                    if vn == "A" and oname == "T" and cand < other_note:
                        break
                    if vn == "A" and oname == "S" and cand > other_note:
                        break
                    if vn == "S" and oname == "T" and cand < other_note:
                        break
                    if vn == "S" and oname == "A" and cand < other_note:
                        break

                    if idx2 > 0 and len(vcur.notes) > (idx2 - 1) and len(ocur.notes) > (idx2 - 1):
                        old_a = vcur.notes[idx2 - 1]
                        old_b = ocur.notes[idx2 - 1]
                        if parallel_or_direct(old_a, old_b, cand, other_note):
                            break
                else:
                    vcur.notes.append(cand)
                    assigned_ok = True
                    break

            if not assigned_ok:
                return False

        return dfs(idx+1)

    for attempt in range(1, max_tries+1):
        progress_cb(int((attempt-1)/max_tries*100))
        log.info(f"🔄 Intento {attempt}/{max_tries}")
        for v in cursors.values():
            v.notes.clear()

        if dfs(0):
            progress_cb(100)
            return {vn: c.notes for vn, c in cursors.items()}

    raise RuntimeError("No se encontró solución con las reglas avanzadas.")

##############################################################################
# 6. PARTITURA Y EXPORTACIÓN
##############################################################################
def score_from_state(
    state: State,
    k: key.Key,
    voice_defs: Dict[str, VoiceDef]
) -> stream.Score:
    s = stream.Score()
    s.insert(0, k)
    s.insert(0, meter.TimeSignature("4/4"))

    cf_part = stream.Part(id="CF")
    cf_part.partName = "Cantus Firmus"
    for midi_val in state.cf:
        cf_part.append(note.Note(midi_val, type="whole"))
    s.insert(0, cf_part)

    dur_map = {1: 4.0, 2: 2.0, 3: 1.0, 4: 2.0}
    for vn in VOICE_NAMES:
        vdef = voice_defs[vn]
        part = stream.Part(id=vn)
        part.partName = f"{vn} ({SPECIES_LABEL.get(vdef.species,'')} especie)"
        for midi_val in state.voices[vn]:
            n = note.Note(midi_val)
            n.quarterLength = dur_map.get(vdef.species, 4.0)
            part.append(n)
        s.insert(0, part)
    return s

def add_metronome(score: stream.Score, beats: int, velocity: int = 110) -> None:
    """ Usar note.Note en lugar de note.Unpitched para evitar errores en MIDI """
    click = stream.Part(id="Click")
    click.insert(0, instrument.Woodblock())

    for _ in range(beats * 4):
        # Reemplazamos Unpitched por Note
        tick = note.Note(56)  # Cowbell GM
        tick.volume.velocity = velocity
        tick.quarterLength = 1.0
        # tick.channel = 9    # si deseas usar canal de percusión
        click.append(tick)

    score.insert(0, click)

def export_files(
    score: stream.Score,
    with_wav: bool = True
) -> Tuple[bytes, bytes, bytes | None]:
    tmpdir = Path(tempfile.mkdtemp(prefix="cp_"))
    midf = tmpdir / "contrapunto.mid"
    xmlf = tmpdir / "contrapunto.musicxml"

    score.write("midi", fp=str(midf))
    score.write("musicxml", fp=str(xmlf))

    wav_bytes = None
    if with_wav and SOUNDFONT and Path(SOUNDFONT).exists():
        try:
            from midi2audio import FluidSynth
            fs = FluidSynth(sound_font=SOUNDFONT)
            wavf = tmpdir / "contrapunto.wav"
            fs.midi_to_audio(str(midf), str(wavf))
            wav_bytes = wavf.read_bytes()
        except Exception as e:
            log.warning(f"No se pudo generar WAV: {e}")

    return midf.read_bytes(), xmlf.read_bytes(), wav_bytes

##############################################################################
# 7. STREAMLIT APP
##############################################################################
st.set_page_config("🎼 Contrapunto PRO Avanzado", layout="wide")
st.title("🎼 Generador de Contrapunto Renacentista – Avanzado")

with st.sidebar:
    st.header("♬ Parámetros principales")
    tonic = st.selectbox("Tónica", ["C", "D", "E", "F", "G", "A", "B"], index=0)
    modo  = st.selectbox("Modo", ["dorian", "phrygian", "lydian", "mixolydian"], index=0)
    key_sel = key.Key(tonic, modo)

    cf_len = st.slider("Compases CF", min_value=6, max_value=24, value=8)
    cf_in  = st.text_area("Cantus Firmus (MIDI, vacío = aleatorio)", placeholder="60,62,64,…")

    st.subheader("Especie por voz")
    default_voices = {
        "T": VoiceDef("T", 1, 48, 67),
        "A": VoiceDef("A", 2, 55, 74),
        "S": VoiceDef("S", 3, 60, 79),
    }
    for vn in VOICE_NAMES:
        sp_choice = st.selectbox(
            f"{vn} (rango {default_voices[vn].midi_lo}-{default_voices[vn].midi_hi})",
            [1,2,3,4],
            index=default_voices[vn].species - 1
        )
        default_voices[vn].species = sp_choice

    seed_val = st.number_input("Seed (0 → aleatoria)", min_value=0, max_value=2**31-1, value=0)
    audio_mode = st.radio("Audio", ["Sin audio", "MIDI→WAV (FluidSynth)"], index=0)
    max_tries = st.slider("Máx intentos Backtracking", 1, 20, 8)

    gen_btn = st.button("🚀 Generar Contrapunto")

progress_bar = st.progress(0, text="Esperando…")
log_placeholder = st.empty()

def ui_progress(pct: int):
    progress_bar.progress(pct, text=f"Progreso {pct}%")

if gen_btn:
    try:
        # 1) CF
        if cf_in.strip():
            cf_list = [int(x.strip()) for x in cf_in.split(",") if x.strip()]
        else:
            cf_list = generate_cantus(
                length=cf_len,
                k=key_sel,
                rng=(48,70),
                seed=seed_val or None
            )
        st.write("Cantus Firmus:", cf_list)

        # 2) Contrapunto
        ui_progress(5)
        voices_gen = generate_counterpoint_pro(
            cf_list,
            default_voices,
            max_tries=max_tries,
            progress_cb=ui_progress
        )
        st.write("### Notas generadas (MIDI)")
        st.json(voices_gen)

        # 3) Score
        state = State(cf_list, voices_gen, lcm_grid=1)
        score = score_from_state(state, key_sel, default_voices)

        # 4) Añadir metrónomo si se desea
        beats_total = cf_len
        if audio_mode == "MIDI→WAV (FluidSynth)":
            add_metronome(score, beats_total)

        # 5) Exportar
        ui_progress(85)
        midi_b, xml_b, wav_b = export_files(score, with_wav=(audio_mode!="Sin audio"))
        ui_progress(100)

        # 6) Descargas
        st.download_button("⬇️ Descargar MIDI", midi_b, file_name="contrapunto.mid")
        st.download_button("⬇️ Descargar MusicXML", xml_b, file_name="contrapunto.musicxml")
        if wav_b:
            st.audio(wav_b, format="audio/wav")

        log_placeholder.success("✅ ¡Contrapunto generado con éxito!")
    except Exception as e:
        log.exception("Error en generación:")
        st.error(f"Error: {e}")
