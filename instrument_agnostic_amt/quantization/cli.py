"""Render & Compare 型ビート量子化の単独 CLI エントリポイント。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..soundfont import RENDER_COMPARE_SPEED_PRESETS, resolve_soundfont_path
from ..velocity.cli.infer_velocity import _resolve_stem_files, _resolve_stem_midi_directory
from .render_compare_defragment import (
    RenderCompareDefragmentConfig,
    defragment_stems_by_render_compare,
)
from .render_compare_quantize import (
    BeatGridInput,
    RenderCompareQuantizeConfig,
    load_beat_grid_from_json,
    load_beat_grid_from_midi,
    quantize_stems_by_render_compare,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize separated-stem MIDI files using beat grid and FluidSynth Render & Compare."
    )
    parser.add_argument(
        "--stem-midis-dir",
        type=Path,
        required=True,
        help="Directory containing per-stem MIDI files (e.g. refined_stem_midis/)",
    )
    parser.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Directory containing separated stem audio files (e.g. bass.wav, drums.wav)",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help="SoundFont (.sf2) used for FluidSynth Render & Compare. Downloads MuseScore_General.sf2 automatically when omitted.",
    )
    parser.add_argument(
        "--speed",
        choices=RENDER_COMPARE_SPEED_PRESETS,
        default="balanced",
        help="Render & Compare speed/quality preset (default: balanced)",
    )
    parser.add_argument(
        "--fluidsynth-executable",
        type=Path,
        default=None,
        help="Path to fluidsynth executable. Searches PATH when omitted.",
    )
    parser.add_argument(
        "--beats-json",
        type=Path,
        default=None,
        help="Path to SheetSage2-Refined *.beats.json file",
    )
    parser.add_argument(
        "--beat-midi",
        type=Path,
        default=None,
        help="Fallback tempo-mapped MIDI file to load beat/downbeat grid from when --beats-json is omitted",
    )
    parser.add_argument(
        "--original-mix-audio",
        type=Path,
        default=None,
        help="Optional original full-mix audio file (also used for SheetSage2-Refined beat tracking when no beat file is given)",
    )
    parser.add_argument(
        "--output-midi",
        type=Path,
        default=None,
        help="Output merged quantized MIDI file path",
    )
    parser.add_argument(
        "--output-stem-dir",
        type=Path,
        default=None,
        help="Optional output directory for per-stem quantized MIDI files",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--no-align-beats",
        action="store_true",
        help="Disable audio onset phase alignment of the beat grid",
    )
    parser.add_argument(
        "--no-verify-full-mix",
        action="store_true",
        help="Skip final before/after full-mix verification rendering",
    )
    parser.add_argument(
        "--no-tempo-map",
        action="store_true",
        help="Do not write aligned beat/meter tempo map into --output-midi",
    )
    parser.add_argument(
        "--defragment",
        action="store_true",
        help="Run Render & Compare sustained-instrument note de-fragmentation before quantization",
    )
    parser.add_argument(
        "--defragment-only",
        action="store_true",
        help="Run Render & Compare note de-fragmentation only (skip beat quantization)",
    )
    parser.add_argument(
        "--defragment-include-plucked",
        action="store_true",
        help="Also include non-sustained instruments (piano, guitar, bass) during de-fragmentation",
    )
    parser.add_argument(
        "--max-midi-melodic-instruments",
        type=int,
        default=15,
        help="Maximum melodic tracks in merged MIDI (16ch MIDI constraint)",
    )
    return parser.parse_args()


def _resolve_beat_grid(args: argparse.Namespace) -> BeatGridInput:
    if args.beats_json is not None:
        return load_beat_grid_from_json(args.beats_json)
    if args.beat_midi is not None:
        return load_beat_grid_from_midi(args.beat_midi)
    if args.original_mix_audio is not None and args.original_mix_audio.is_file():
        import soundfile as sf
        import torch

        from ..beat_chord.sheetsage import load_sheetsage_model, predict_sheetsage_beat_grid

        device = "cuda" if torch.cuda.is_available() else "cpu"
        waveform, sample_rate = sf.read(str(args.original_mix_audio), dtype="float32", always_2d=True)
        model = load_sheetsage_model(device)
        sheetsage_grid, _ = predict_sheetsage_beat_grid(model, waveform.mean(axis=1), int(sample_rate))
        return BeatGridInput(
            beat_times=sheetsage_grid.beat_times,
            downbeat_times=sheetsage_grid.downbeat_times,
            meter_segments=sheetsage_grid.meter_segments,
        )
    raise ValueError("Specify --beats-json, --beat-midi, or --original-mix-audio to provide a beat grid.")


def main() -> None:
    args = parse_args()
    resolved_soundfont = resolve_soundfont_path(args.soundfont)
    stem_midis = _resolve_stem_midi_directory(args.stem_midis_dir)
    stem_audios = _resolve_stem_files(args.stems_dir)

    defrag_report: dict[str, object] | None = None
    if args.defragment or args.defragment_only:
        defrag_stem_dir = (
            args.output_stem_dir
            if args.defragment_only and args.output_stem_dir is not None
            else (args.stem_midis_dir / "defragmented_stem_midis")
        )
        defrag_output_midi = (
            (args.output_midi or (args.stem_midis_dir / "merged_rc_defragmented.mid"))
            if args.defragment_only
            else None
        )
        base_defrag_config = RenderCompareDefragmentConfig.for_speed_preset(
            resolved_soundfont,
            speed=args.speed,
            fluidsynth_executable=args.fluidsynth_executable,
            max_melodic_instruments=int(args.max_midi_melodic_instruments),
        )
        defrag_report = defragment_stems_by_render_compare(
            stem_midis=stem_midis,
            stem_audios=stem_audios,
            config=RenderCompareDefragmentConfig(
                soundfont_path=base_defrag_config.soundfont_path,
                fluidsynth_executable=base_defrag_config.fluidsynth_executable,
                sustained_only=not args.defragment_include_plucked,
                max_melodic_instruments=base_defrag_config.max_melodic_instruments,
                render_active_windows_only=base_defrag_config.render_active_windows_only,
            ),
            output_midi_path=defrag_output_midi,
            output_stem_dir=defrag_stem_dir,
        )
        if args.defragment_only:
            if args.output_json is not None:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(json.dumps(defrag_report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(defrag_report, ensure_ascii=False, indent=2))
            print(f"Successfully generated Render & Compare defragmented MIDI: {defrag_report['output_midi_path']}")
            return
        stem_midis = {name: Path(path) for name, path in defrag_report["output_stem_paths"].items()}

    beat_grid = _resolve_beat_grid(args)

    output_midi = args.output_midi or (args.stem_midis_dir / "merged_rc_quantized.mid")
    base_quantize_config = RenderCompareQuantizeConfig.for_speed_preset(
        resolved_soundfont,
        speed=args.speed,
        fluidsynth_executable=args.fluidsynth_executable,
        max_melodic_instruments=int(args.max_midi_melodic_instruments),
        write_tempo_map=not args.no_tempo_map,
    )
    report = quantize_stems_by_render_compare(
        stem_midis=stem_midis,
        stem_audios=stem_audios,
        beat_grid=beat_grid,
        config=RenderCompareQuantizeConfig(
            soundfont_path=base_quantize_config.soundfont_path,
            fluidsynth_executable=base_quantize_config.fluidsynth_executable,
            align_beats_to_audio=not args.no_align_beats,
            verify_full_mix=False if args.no_verify_full_mix else base_quantize_config.verify_full_mix,
            write_tempo_map=base_quantize_config.write_tempo_map,
            max_melodic_instruments=base_quantize_config.max_melodic_instruments,
            adaptive_candidates=base_quantize_config.adaptive_candidates,
        ),
        original_mix_audio=args.original_mix_audio,
        output_midi_path=output_midi,
        output_stem_dir=args.output_stem_dir,
    )
    if defrag_report is not None:
        report["defragment"] = defrag_report
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Successfully generated Render & Compare quantized MIDI: {report['output_midi_path']}")


if __name__ == "__main__":
    main()
