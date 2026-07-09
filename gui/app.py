from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gradio as gr

from . import backend


APP_CSS = """
.dggr-shell {max-width: 1500px; margin: 0 auto;}
.dggr-hero {padding: 10px 0 14px 0;}
.dggr-hero h1 {font-size: 2.2rem; margin-bottom: 0.3rem;}
.dggr-soft {border: 1px solid rgba(15,118,110,0.15); border-radius: 16px; padding: 14px;}
.dggr-muted {opacity: 0.78;}
.dggr-preset-scroll {height: 240px; overflow-y: auto !important; overscroll-behavior: contain; border: 1px solid rgba(15,118,110,0.15); border-radius: 12px; padding: 10px;}
.dggr-source-chip {margin-top: 6px;}
"""


def _example_choices() -> List[Tuple[str, str]]:
    return [(Path(p).name, p) for p in backend.catalog_snapshot()["example_audio"]]


def _resolve_audio(uploaded: str | None, example: str | None) -> str:
    path = uploaded or example
    if not path:
        raise gr.Error("Upload an audio file or choose an example clip first.")
    return path


def _resolve_audio_with_source(uploaded: str | None, example: str | None) -> tuple[str, str]:
    if uploaded:
        return uploaded, f"Uploaded clip: {Path(uploaded).name}"
    if example:
        return example, f"Preset clip: {Path(example).name}"
    raise gr.Error("Upload an audio file or choose an example clip first.")


def _describe_source(uploaded: str | None, example: str | None) -> str:
    if uploaded:
        return f"**Selected source**: uploaded clip `{Path(uploaded).name}`"
    if example:
        return f"**Selected source**: preset clip `{Path(example).name}`"
    return "**Selected source**: none"


def _compare_target_choices(codec_run: str, diffusion_run: str):
    codec_genres = set(backend.genres_for_codec_run(codec_run))
    diffusion_genres = set(backend.genres_for_diffusion_run(diffusion_run))
    choices = sorted(codec_genres & diffusion_genres)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def _paired_compare_targets(codec_run: str, diffusion_run: str):
    update = _compare_target_choices(codec_run, diffusion_run)
    return update, update


def _codec_checkpoint_update(run_name: str):
    choices, value = backend.codec_checkpoint_choices(run_name)
    return gr.update(choices=choices, value=value)


def _diffusion_checkpoint_update(run_name: str):
    choices, value = backend.diffusion_checkpoint_choices(run_name)
    return gr.update(choices=choices, value=value)


def _single_codec_genres(run_name: str):
    choices = backend.genres_for_codec_run(run_name)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def _single_diffusion_genres(run_name: str):
    choices = backend.genres_for_diffusion_run(run_name)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value), gr.update(choices=choices, value=value)


def _single_diffusion_target(run_name: str):
    choices = backend.genres_for_diffusion_run(run_name)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def _codec_longform_target(run_name: str):
    choices = backend.genres_for_codec_run(run_name)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def _longform_genres(run_name: str):
    choices = backend.genres_for_diffusion_run(run_name)
    if not choices:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    return gr.update(choices=choices, value=choices[0]), gr.update(choices=choices, value=choices[-1])


def _refresh_catalog(device: str):
    snap = backend.catalog_snapshot()
    sys_md, codec_df, diffusion_df = backend.system_snapshot(device)
    compare_choices = _compare_target_choices(snap["codec_default"], snap["diffusion_default"])
    long_compare_codec_target, long_compare_diff_target = _paired_compare_targets(snap["codec_default"], snap["diffusion_default"])
    codec_genres = _single_codec_genres(snap["codec_default"])
    codec_long_genres = _codec_longform_target(snap["codec_default"])
    diff_target = _single_diffusion_target(snap["diffusion_default"])
    long_source, long_target = _longform_genres(snap["diffusion_default"])
    long_compare_source, _ignored = _longform_genres(snap["diffusion_default"])
    compare_codec_ckpt = _codec_checkpoint_update(snap["codec_default"])
    codec_ckpt = _codec_checkpoint_update(snap["codec_default"])
    codec_long_ckpt = _codec_checkpoint_update(snap["codec_default"])
    long_compare_codec_ckpt = _codec_checkpoint_update(snap["codec_default"])
    compare_diff_ckpt = _diffusion_checkpoint_update(snap["diffusion_default"])
    diffusion_ckpt = _diffusion_checkpoint_update(snap["diffusion_default"])
    long_ckpt = _diffusion_checkpoint_update(snap["diffusion_default"])
    long_compare_diff_ckpt = _diffusion_checkpoint_update(snap["diffusion_default"])
    example_choices = _example_choices()
    example_value = example_choices[0][1] if example_choices else None
    return (
        gr.update(choices=snap["codec_runs"], value=snap["codec_default"]),
        gr.update(choices=snap["codec_runs"], value=snap["codec_default"]),
        gr.update(choices=snap["codec_runs"], value=snap["codec_default"]),
        gr.update(choices=snap["codec_runs"], value=snap["codec_default"]),
        compare_codec_ckpt,
        codec_ckpt,
        codec_long_ckpt,
        long_compare_codec_ckpt,
        gr.update(choices=snap["diffusion_runs"], value=snap["diffusion_default"]),
        gr.update(choices=snap["diffusion_runs"], value=snap["diffusion_default"]),
        gr.update(choices=snap["diffusion_runs"], value=snap["diffusion_default"]),
        gr.update(choices=snap["diffusion_runs"], value=snap["diffusion_default"]),
        compare_diff_ckpt,
        diffusion_ckpt,
        long_ckpt,
        long_compare_diff_ckpt,
        gr.update(choices=example_choices, value=example_value),
        _describe_source(None, example_value),
        compare_choices,
        codec_genres,
        diff_target,
        codec_long_genres,
        long_source,
        long_target,
        long_compare_codec_target,
        long_compare_source,
        long_compare_diff_target,
        sys_md,
        codec_df,
        diffusion_df,
        backend.get_terminal_log(),
    )


def _clear_model_cache():
    backend.SESSION_CACHE.clear()
    terminal = backend.clear_terminal_log()
    terminal = backend.append_terminal_log("Model cache cleared.")
    return "### Cache cleared\n\nModel sessions and cached vocoders were released.", terminal


def _terminal_text() -> str:
    return backend.get_terminal_log()


def _analyze(uploaded: str | None, example: str | None):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Analyzing selected audio input. {source_label}")
    result = backend.analyze_audio_for_ui(path)
    terminal = backend.append_terminal_log("Audio analysis ready.")
    return f"{result[0]}\n\n**Source**: {source_label}", result[1], terminal


def _run_codec(uploaded, example, codec_run, codec_checkpoint, target_genre, style_mode, mix_alpha, start_sec, seed, device):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing codec inference. {source_label}")
    result = backend.run_codec_job(
        path,
        codec_run,
        codec_checkpoint,
        target_genre,
        style_mode,
        mix_alpha,
        start_sec,
        seed,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], backend.get_terminal_log()


def _run_diffusion(uploaded, example, diffusion_run, diffusion_checkpoint, target_genre, start_sec, clip_seconds, guidance_scale, ddim_steps, eta, seed, device):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing diffusion inference. {source_label}")
    result = backend.run_diffusion_job(
        path,
        diffusion_run,
        diffusion_checkpoint,
        target_genre,
        start_sec,
        clip_seconds,
        guidance_scale,
        ddim_steps,
        eta,
        seed,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], backend.get_terminal_log()


def _run_real_music(uploaded, example, checkpoint, target_genre, seconds, chunk_seconds, overlap_seconds, device):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing real-music transfer. {source_label}")
    result = backend.run_real_music_job(
        path,
        checkpoint,
        target_genre,
        seconds,
        chunk_seconds,
        overlap_seconds,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], backend.get_terminal_log()


def _run_compare(uploaded, example, codec_run, codec_checkpoint, diffusion_run, diffusion_checkpoint, target_genre, start_sec, codec_style_mode, codec_mix_alpha, diffusion_seconds, guidance_scale, ddim_steps, seed, device):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing side-by-side comparison. {source_label}")
    result = backend.run_compare_job(
        path,
        codec_run,
        codec_checkpoint,
        diffusion_run,
        diffusion_checkpoint,
        target_genre,
        start_sec,
        codec_style_mode,
        codec_mix_alpha,
        diffusion_seconds,
        guidance_scale,
        ddim_steps,
        seed,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], result[6], backend.get_terminal_log()


def _run_longform(uploaded, example, diffusion_run, diffusion_checkpoint, source_genre, target_genre, source_start_sec, source_seconds, chunk_seconds, overlap_seconds, t_start, reanchor_every, reanchor_t_start, guidance_scale, style_strength, prefix_blend, source_prefix_blend, source_mel_blend, hf_source_blend, mel_time_smooth, mel_freq_smooth, assemble_domain, ddim_steps, seed, device):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing long-form coherence run. {source_label}")
    result = backend.run_longform_job(
        path,
        diffusion_run,
        diffusion_checkpoint,
        source_genre,
        target_genre,
        source_start_sec,
        source_seconds,
        chunk_seconds,
        overlap_seconds,
        t_start,
        reanchor_every,
        reanchor_t_start,
        guidance_scale,
        style_strength,
        prefix_blend,
        source_prefix_blend,
        source_mel_blend,
        hf_source_blend,
        mel_time_smooth,
        mel_freq_smooth,
        assemble_domain,
        ddim_steps,
        seed,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], result[6], backend.get_terminal_log()


def _run_codec_longform(uploaded, example, codec_run, codec_checkpoint, target_genre, style_mode, mix_alpha, source_start_sec, source_seconds, chunk_seconds, overlap_seconds, seed, device):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing codec long-form run. {source_label}")
    result = backend.run_codec_longform_job(
        path,
        codec_run,
        codec_checkpoint,
        target_genre,
        style_mode,
        mix_alpha,
        source_start_sec,
        source_seconds,
        chunk_seconds,
        overlap_seconds,
        seed,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], result[6], backend.get_terminal_log()


def _run_longform_compare(
    uploaded,
    example,
    codec_run,
    codec_checkpoint,
    codec_target_genre,
    codec_style_mode,
    codec_mix_alpha,
    diffusion_run,
    diffusion_checkpoint,
    diffusion_source_genre,
    diffusion_target_genre,
    source_start_sec,
    source_seconds,
    codec_chunk_seconds,
    codec_overlap_seconds,
    diffusion_chunk_seconds,
    diffusion_overlap_seconds,
    t_start,
    reanchor_every,
    reanchor_t_start,
    guidance_scale,
    style_strength,
    prefix_blend,
    source_prefix_blend,
    source_mel_blend,
    hf_source_blend,
    mel_time_smooth,
    mel_freq_smooth,
    assemble_domain,
    ddim_steps,
    seed,
    device,
):
    path, source_label = _resolve_audio_with_source(uploaded, example)
    backend.clear_terminal_log()
    backend.append_terminal_log(f"Preparing long-form codec vs diffusion comparison. {source_label}")
    result = backend.run_longform_compare_job(
        path,
        codec_run,
        codec_checkpoint,
        codec_target_genre,
        codec_style_mode,
        codec_mix_alpha,
        diffusion_run,
        diffusion_checkpoint,
        diffusion_source_genre,
        diffusion_target_genre,
        source_start_sec,
        source_seconds,
        codec_chunk_seconds,
        codec_overlap_seconds,
        diffusion_chunk_seconds,
        diffusion_overlap_seconds,
        t_start,
        reanchor_every,
        reanchor_t_start,
        guidance_scale,
        style_strength,
        prefix_blend,
        source_prefix_blend,
        source_mel_blend,
        hf_source_blend,
        mel_time_smooth,
        mel_freq_smooth,
        assemble_domain,
        ddim_steps,
        seed,
        device,
        log_callback=backend.append_terminal_log,
    )
    return result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], backend.get_terminal_log()


def build_app() -> gr.Blocks:
    snap = backend.catalog_snapshot()
    sys_md, codec_df, diffusion_df = backend.system_snapshot("auto")
    compare_choices = sorted(set(backend.genres_for_codec_run(snap["codec_default"])) & set(backend.genres_for_diffusion_run(snap["diffusion_default"])))
    default_compare_genre = compare_choices[0] if compare_choices else None
    codec_genres = backend.genres_for_codec_run(snap["codec_default"])
    diffusion_genres = backend.genres_for_diffusion_run(snap["diffusion_default"])
    compare_codec_ckpts, compare_codec_ckpt_default = backend.codec_checkpoint_choices(snap["codec_default"])
    compare_diff_ckpts, compare_diff_ckpt_default = backend.diffusion_checkpoint_choices(snap["diffusion_default"])
    codec_ckpts, codec_ckpt_default = backend.codec_checkpoint_choices(snap["codec_default"])
    diffusion_ckpts, diffusion_ckpt_default = backend.diffusion_checkpoint_choices(snap["diffusion_default"])
    long_ckpts, long_ckpt_default = backend.diffusion_checkpoint_choices(snap["diffusion_default"])
    codec_long_ckpts, codec_long_ckpt_default = backend.codec_checkpoint_choices(snap["codec_default"])
    long_compare_codec_ckpts, long_compare_codec_ckpt_default = backend.codec_checkpoint_choices(snap["codec_default"])
    long_compare_diff_ckpts, long_compare_diff_ckpt_default = backend.diffusion_checkpoint_choices(snap["diffusion_default"])
    real_music_ckpts, real_music_ckpt_default = backend.real_music_checkpoint_choices()
    real_music_genres = backend.real_music_genres()

    with gr.Blocks(title="DGGR Inference Studio") as demo:
        gr.Markdown(
            """
            <div class="dggr-shell dggr-hero">
            <h1>DGGR Inference Studio</h1>
            <p class="dggr-muted">Local GUI for short-form and long-form DGGR experiments using the actual CMPUT 414 repo checkpoints. Codec and diffusion are split out explicitly so you can compare short-form quality, long-form stability, and checkpoint behavior without guessing which path is running.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Upload source audio",
                    type="filepath",
                    sources=["upload"],
                    format="wav",
                )
            with gr.Column(scale=1):
                gr.Markdown("**Preset clips**")
                example_input = gr.Radio(
                    label="Choose a preset clip",
                    choices=_example_choices(),
                    value=snap["example_audio"][0] if snap["example_audio"] else None,
                    elem_classes=["dggr-preset-scroll"],
                )
                device_choice = gr.Radio(
                    label="Execution device",
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                )
                analyze_btn = gr.Button("Analyze Input", variant="secondary")

        source_status = gr.Markdown(_describe_source(None, snap["example_audio"][0] if snap["example_audio"] else None))

        with gr.Row():
            audio_info_md = gr.Markdown("### Audio status\n\nUpload a file or choose an example clip.")
            audio_info_plot = gr.Image(label="Input preview", type="filepath")

        with gr.Tabs():
            with gr.Tab("Short-Form Compare"):
                gr.Markdown("Compare one codec checkpoint against one diffusion checkpoint on the same short excerpt.")
                with gr.Row():
                    codec_run = gr.Dropdown(label="Codec run", choices=snap["codec_runs"], value=snap["codec_default"])
                    compare_codec_checkpoint = gr.Dropdown(label="Codec checkpoint", choices=compare_codec_ckpts, value=compare_codec_ckpt_default)
                    diffusion_run = gr.Dropdown(label="Diffusion run", choices=snap["diffusion_runs"], value=snap["diffusion_default"])
                    compare_diffusion_checkpoint = gr.Dropdown(label="Diffusion checkpoint", choices=compare_diff_ckpts, value=compare_diff_ckpt_default)
                    compare_target = gr.Dropdown(label="Target genre", choices=compare_choices, value=default_compare_genre)
                with gr.Row():
                    compare_start = gr.Slider(label="Start offset (seconds)", minimum=0, maximum=30, step=0.25, value=0.0)
                    compare_diffusion_seconds = gr.Slider(label="Diffusion clip length (seconds)", minimum=1.0, maximum=5.0, step=0.25, value=3.0)
                    compare_guidance = gr.Slider(label="Diffusion guidance", minimum=1.0, maximum=5.0, step=0.1, value=2.0)
                    compare_steps = gr.Slider(label="DDIM steps", minimum=10, maximum=100, step=5, value=50)
                with gr.Row():
                    compare_style_mode = gr.Radio(label="Codec style conditioning", choices=["centroid", "exemplar", "mix"], value="mix")
                    compare_mix_alpha = gr.Slider(label="Codec centroid weight", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                    compare_seed = gr.Number(label="Seed", value=328, precision=0)
                compare_btn = gr.Button("Run Side-by-Side Comparison", variant="primary")
                compare_md = gr.Markdown()
                compare_metrics = gr.Dataframe(label="Compare metrics", interactive=False)
                with gr.Row():
                    compare_src_audio = gr.Audio(label="Source clip")
                    compare_codec_audio = gr.Audio(label="Codec result")
                    compare_diff_audio = gr.Audio(label="Diffusion result")
                compare_plot = gr.Image(label="Comparison figure", type="filepath")
                compare_zip = gr.File(label="Download result bundle")

            with gr.Tab("Codec Explorer"):
                gr.Markdown("Run the source-faithful codec branch on a short excerpt.")
                with gr.Row():
                    codec_run_single = gr.Dropdown(label="Codec run", choices=snap["codec_runs"], value=snap["codec_default"])
                    codec_checkpoint_single = gr.Dropdown(label="Checkpoint", choices=codec_ckpts, value=codec_ckpt_default)
                    codec_target = gr.Dropdown(label="Target genre", choices=codec_genres, value=codec_genres[0] if codec_genres else None)
                    codec_style_mode = gr.Radio(label="Style mode", choices=["centroid", "exemplar", "mix"], value="mix")
                with gr.Row():
                    codec_mix_alpha = gr.Slider(label="Centroid/exemplar blend", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                    codec_start = gr.Slider(label="Start offset (seconds)", minimum=0, maximum=30, step=0.25, value=0.0)
                    codec_seed = gr.Number(label="Seed", value=328, precision=0)
                codec_btn = gr.Button("Run Codec Inference", variant="primary")
                codec_md = gr.Markdown()
                codec_metrics = gr.Dataframe(label="Codec metrics", interactive=False)
                with gr.Row():
                    codec_src_audio = gr.Audio(label="Source clip")
                    codec_out_audio = gr.Audio(label="Codec output")
                codec_plot = gr.Image(label="Codec preview", type="filepath")
                codec_zip = gr.File(label="Download result bundle")

            with gr.Tab("Diffusion Explorer"):
                gr.Markdown("Run the diffusion branch on a short excerpt with full sampling controls.")
                with gr.Row():
                    diffusion_run_single = gr.Dropdown(label="Diffusion run", choices=snap["diffusion_runs"], value=snap["diffusion_default"])
                    diffusion_checkpoint_single = gr.Dropdown(label="Checkpoint", choices=diffusion_ckpts, value=diffusion_ckpt_default)
                    diffusion_target = gr.Dropdown(label="Target genre", choices=diffusion_genres, value=diffusion_genres[0] if diffusion_genres else None)
                with gr.Row():
                    diffusion_start = gr.Slider(label="Start offset (seconds)", minimum=0, maximum=30, step=0.25, value=0.0)
                    diffusion_seconds = gr.Slider(label="Clip length (seconds)", minimum=1.0, maximum=5.0, step=0.25, value=3.0)
                    diffusion_guidance = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, step=0.1, value=2.0)
                    diffusion_steps = gr.Slider(label="DDIM steps", minimum=10, maximum=100, step=5, value=50)
                with gr.Row():
                    diffusion_eta = gr.Slider(label="DDIM eta", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                    diffusion_seed = gr.Number(label="Seed", value=328, precision=0)
                diffusion_btn = gr.Button("Run Diffusion Inference", variant="primary")
                diffusion_md = gr.Markdown()
                diffusion_metrics = gr.Dataframe(label="Diffusion metrics", interactive=False)
                with gr.Row():
                    diffusion_src_audio = gr.Audio(label="Source clip")
                    diffusion_out_audio = gr.Audio(label="Diffusion output")
                diffusion_plot = gr.Image(label="Diffusion preview", type="filepath")
                diffusion_zip = gr.File(label="Download result bundle")

            with gr.Tab("Real-Music Transfer"):
                gr.Markdown("Run the real-music discovered-style model trained from the downloaded Spotify-derived audio folder.")
                with gr.Row():
                    real_music_checkpoint = gr.Dropdown(label="Real-music checkpoint", choices=real_music_ckpts, value=real_music_ckpt_default)
                    real_music_target = gr.Dropdown(
                        label="Target discovered family",
                        choices=real_music_genres,
                        value=real_music_genres[0] if real_music_genres else None,
                    )
                with gr.Row():
                    real_music_seconds = gr.Slider(label="Render length (seconds)", minimum=3.0, maximum=90.0, step=1.0, value=24.0)
                    real_music_chunk = gr.Slider(label="Chunk size (seconds)", minimum=3.0, maximum=8.0, step=0.5, value=3.0)
                    real_music_overlap = gr.Slider(label="Overlap (seconds)", minimum=0.25, maximum=2.0, step=0.25, value=0.5)
                real_music_btn = gr.Button("Run Real-Music Transfer", variant="primary")
                real_music_md = gr.Markdown()
                real_music_metrics = gr.Dataframe(label="Real-music metrics", interactive=False)
                with gr.Row():
                    real_music_src_audio = gr.Audio(label="Source excerpt")
                    real_music_out_audio = gr.Audio(label="Real-music output")
                real_music_plot = gr.Image(label="Real-music preview", type="filepath")
                real_music_zip = gr.File(label="Download real-music bundle")
                real_music_log = gr.Textbox(label="Real-music log", lines=12)

            with gr.Tab("Codec Long-Form"):
                gr.Markdown("Chunk the source across a longer excerpt and run codec transfer on every chunk, then assemble it with overlap-add so you can hear how codec errors compound.")
                with gr.Row():
                    codec_long_run = gr.Dropdown(label="Codec run", choices=snap["codec_runs"], value=snap["codec_default"])
                    codec_long_checkpoint = gr.Dropdown(label="Checkpoint", choices=codec_long_ckpts, value=codec_long_ckpt_default)
                    codec_long_target = gr.Dropdown(label="Target genre", choices=codec_genres, value=codec_genres[0] if codec_genres else None)
                    codec_long_style_mode = gr.Radio(label="Style mode", choices=["centroid", "exemplar", "mix"], value="mix")
                with gr.Row():
                    codec_long_mix_alpha = gr.Slider(label="Centroid/exemplar blend", minimum=0.0, maximum=1.0, step=0.05, value=0.35)
                    codec_long_start = gr.Slider(label="Source start (seconds)", minimum=0, maximum=120, step=0.5, value=0.0)
                    codec_long_seconds = gr.Slider(label="Source length (seconds)", minimum=10, maximum=120, step=5, value=45)
                    codec_long_seed = gr.Number(label="Seed", value=328, precision=0)
                with gr.Row():
                    codec_long_chunk = gr.Slider(label="Chunk size (seconds)", minimum=3.0, maximum=8.0, step=0.25, value=5.0)
                    codec_long_overlap = gr.Slider(label="Overlap (seconds)", minimum=0.25, maximum=1.5, step=0.05, value=0.5)
                codec_long_btn = gr.Button("Run Codec Long-Form", variant="primary")
                codec_long_md = gr.Markdown()
                codec_long_metrics = gr.Dataframe(label="Codec long-form metrics", interactive=False)
                with gr.Row():
                    codec_long_src_audio = gr.Audio(label="Source excerpt")
                    codec_long_out_audio = gr.Audio(label="Codec long-form output")
                codec_long_plot = gr.Image(label="Codec long-form preview", type="filepath")
                codec_long_zip = gr.File(label="Download codec long-form bundle")
                codec_long_log = gr.Textbox(label="Codec chunk log", lines=12)

            with gr.Tab("Diffusion Long-Form"):
                gr.Markdown("Run the Lab 4 long-form diffusion/coherence pipeline directly with checkpoint-specific controls.")
                with gr.Row():
                    long_run = gr.Dropdown(label="Diffusion run", choices=snap["diffusion_runs"], value=snap["diffusion_default"])
                    long_checkpoint = gr.Dropdown(label="Checkpoint", choices=long_ckpts, value=long_ckpt_default)
                    long_source_genre = gr.Dropdown(label="Source genre label", choices=diffusion_genres, value=diffusion_genres[0] if diffusion_genres else None)
                    long_target_genre = gr.Dropdown(label="Target genre label", choices=diffusion_genres, value=diffusion_genres[-1] if diffusion_genres else None)
                with gr.Row():
                    long_start = gr.Slider(label="Source start (seconds)", minimum=0, maximum=120, step=0.5, value=0.0)
                    long_seconds = gr.Slider(label="Source length (seconds)", minimum=10, maximum=120, step=5, value=30)
                    long_chunk = gr.Slider(label="Chunk size (seconds)", minimum=2.0, maximum=5.0, step=0.25, value=3.0)
                    long_overlap = gr.Slider(label="Overlap (seconds)", minimum=0.25, maximum=1.5, step=0.05, value=0.5)
                with gr.Accordion("Advanced continuity controls", open=False):
                    with gr.Row():
                        long_t_start = gr.Slider(label="Initial noise step", minimum=50, maximum=600, step=10, value=240)
                        long_reanchor_every = gr.Slider(label="Re-anchor every N chunks", minimum=1, maximum=24, step=1, value=4)
                        long_reanchor_t = gr.Slider(label="Re-anchor noise step", minimum=50, maximum=400, step=10, value=160)
                        long_ddim_steps = gr.Slider(label="DDIM steps", minimum=10, maximum=100, step=5, value=50)
                    with gr.Row():
                        long_guidance = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, step=0.1, value=1.75)
                        long_style_strength = gr.Slider(label="Style strength", minimum=0.0, maximum=1.0, step=0.05, value=0.60)
                        long_prefix = gr.Slider(label="Prefix blend", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                        long_source_prefix = gr.Slider(label="Source prefix blend", minimum=0.0, maximum=1.0, step=0.05, value=0.45)
                    with gr.Row():
                        long_source_mel = gr.Slider(label="Source mel blend", minimum=0.0, maximum=1.0, step=0.05, value=0.10)
                        long_hf_source = gr.Slider(label="HF source blend", minimum=0.0, maximum=1.0, step=0.05, value=0.18)
                        long_mel_time = gr.Slider(label="Mel time smoothing", minimum=0, maximum=15, step=1, value=3)
                        long_mel_freq = gr.Slider(label="Mel freq smoothing", minimum=0, maximum=7, step=1, value=0)
                    with gr.Row():
                        long_assemble = gr.Radio(label="Assembly domain", choices=["mel", "wave"], value="mel")
                        long_seed = gr.Number(label="Seed", value=328, precision=0)
                long_btn = gr.Button("Run Long-Form Transfer", variant="primary")
                long_md = gr.Markdown()
                long_metrics = gr.Dataframe(label="Long-form metrics", interactive=False)
                with gr.Row():
                    long_src_audio = gr.Audio(label="Source excerpt")
                    long_out_audio = gr.Audio(label="Long-form output")
                long_plot = gr.Image(label="Long-form preview", type="filepath")
                long_zip = gr.File(label="Download result bundle")
                long_log = gr.Textbox(label="Runner log tail", lines=16)

            with gr.Tab("Long-Form Compare"):
                gr.Markdown("Run codec long-form and diffusion long-form on the same source excerpt so you can hear realism versus stability tradeoffs directly.")
                with gr.Row():
                    long_compare_codec_run = gr.Dropdown(label="Codec run", choices=snap["codec_runs"], value=snap["codec_default"])
                    long_compare_codec_checkpoint = gr.Dropdown(label="Codec checkpoint", choices=long_compare_codec_ckpts, value=long_compare_codec_ckpt_default)
                    long_compare_codec_target = gr.Dropdown(label="Codec target genre", choices=compare_choices, value=default_compare_genre)
                    long_compare_codec_style_mode = gr.Radio(label="Codec style mode", choices=["centroid", "exemplar", "mix"], value="mix")
                with gr.Row():
                    long_compare_diff_run = gr.Dropdown(label="Diffusion run", choices=snap["diffusion_runs"], value=snap["diffusion_default"])
                    long_compare_diff_checkpoint = gr.Dropdown(label="Diffusion checkpoint", choices=long_compare_diff_ckpts, value=long_compare_diff_ckpt_default)
                    long_compare_diff_source = gr.Dropdown(label="Diffusion source genre", choices=diffusion_genres, value=diffusion_genres[0] if diffusion_genres else None)
                    long_compare_diff_target = gr.Dropdown(label="Diffusion target genre", choices=compare_choices, value=default_compare_genre)
                with gr.Row():
                    long_compare_start = gr.Slider(label="Source start (seconds)", minimum=0, maximum=120, step=0.5, value=0.0)
                    long_compare_seconds = gr.Slider(label="Source length (seconds)", minimum=10, maximum=120, step=5, value=45)
                    long_compare_seed = gr.Number(label="Seed", value=328, precision=0)
                with gr.Row():
                    long_compare_codec_mix_alpha = gr.Slider(label="Codec centroid/exemplar blend", minimum=0.0, maximum=1.0, step=0.05, value=0.35)
                    long_compare_codec_chunk = gr.Slider(label="Codec chunk size (seconds)", minimum=3.0, maximum=8.0, step=0.25, value=5.0)
                    long_compare_codec_overlap = gr.Slider(label="Codec overlap (seconds)", minimum=0.25, maximum=1.5, step=0.05, value=0.5)
                with gr.Accordion("Diffusion long-form controls", open=False):
                    with gr.Row():
                        long_compare_diff_chunk = gr.Slider(label="Diffusion chunk size (seconds)", minimum=2.0, maximum=5.0, step=0.25, value=3.0)
                        long_compare_diff_overlap = gr.Slider(label="Diffusion overlap (seconds)", minimum=0.25, maximum=1.5, step=0.05, value=0.5)
                        long_compare_t_start = gr.Slider(label="Initial noise step", minimum=50, maximum=600, step=10, value=240)
                        long_compare_reanchor_every = gr.Slider(label="Re-anchor every N chunks", minimum=1, maximum=24, step=1, value=4)
                    with gr.Row():
                        long_compare_reanchor_t = gr.Slider(label="Re-anchor noise step", minimum=50, maximum=400, step=10, value=160)
                        long_compare_guidance = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, step=0.1, value=1.75)
                        long_compare_style_strength = gr.Slider(label="Style strength", minimum=0.0, maximum=1.0, step=0.05, value=0.60)
                        long_compare_prefix = gr.Slider(label="Prefix blend", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                    with gr.Row():
                        long_compare_source_prefix = gr.Slider(label="Source prefix blend", minimum=0.0, maximum=1.0, step=0.05, value=0.45)
                        long_compare_source_mel = gr.Slider(label="Source mel blend", minimum=0.0, maximum=1.0, step=0.05, value=0.10)
                        long_compare_hf_source = gr.Slider(label="HF source blend", minimum=0.0, maximum=1.0, step=0.05, value=0.18)
                        long_compare_ddim_steps = gr.Slider(label="DDIM steps", minimum=10, maximum=100, step=5, value=50)
                    with gr.Row():
                        long_compare_mel_time = gr.Slider(label="Mel time smoothing", minimum=0, maximum=15, step=1, value=3)
                        long_compare_mel_freq = gr.Slider(label="Mel freq smoothing", minimum=0, maximum=7, step=1, value=0)
                        long_compare_assemble = gr.Radio(label="Assembly domain", choices=["mel", "wave"], value="mel")
                long_compare_btn = gr.Button("Run Long-Form Codec vs Diffusion", variant="primary")
                long_compare_md = gr.Markdown()
                long_compare_metrics = gr.Dataframe(label="Long-form compare metrics", interactive=False)
                with gr.Row():
                    long_compare_src_audio = gr.Audio(label="Shared source excerpt")
                    long_compare_codec_audio = gr.Audio(label="Codec long-form output")
                    long_compare_diff_audio = gr.Audio(label="Diffusion long-form output")
                with gr.Row():
                    long_compare_codec_plot = gr.Image(label="Codec long-form preview", type="filepath")
                    long_compare_diff_plot = gr.Image(label="Diffusion long-form preview", type="filepath")
                with gr.Row():
                    long_compare_codec_zip = gr.File(label="Codec long-form bundle")
                    long_compare_diff_zip = gr.File(label="Diffusion long-form bundle")
                long_compare_log = gr.Textbox(label="Long-form compare log", lines=16)

            with gr.Tab("Run Browser + System"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh run catalog", variant="secondary")
                    clear_cache_btn = gr.Button("Clear loaded model cache", variant="secondary")
                system_md = gr.Markdown(sys_md)
                codec_table = gr.Dataframe(value=codec_df, interactive=False, label="Codec runs")
                diffusion_table = gr.Dataframe(value=diffusion_df, interactive=False, label="Diffusion runs")
                real_music_table = gr.Dataframe(value=backend.real_music_runs_table(), interactive=False, label="Real-music run")

        terminal_box = gr.Textbox(
            label="Runtime terminal",
            value=backend.get_terminal_log(),
            lines=12,
            max_lines=20,
            autoscroll=True,
            interactive=False,
        )
        terminal_timer = gr.Timer(1.0)

        audio_input.change(_describe_source, [audio_input, example_input], [source_status])
        example_input.change(_describe_source, [audio_input, example_input], [source_status])
        analyze_btn.click(_analyze, [audio_input, example_input], [audio_info_md, audio_info_plot, terminal_box])
        codec_run.change(_compare_target_choices, [codec_run, diffusion_run], [compare_target])
        diffusion_run.change(_compare_target_choices, [codec_run, diffusion_run], [compare_target])
        codec_run.change(_codec_checkpoint_update, [codec_run], [compare_codec_checkpoint])
        codec_run_single.change(_codec_checkpoint_update, [codec_run_single], [codec_checkpoint_single])
        codec_long_run.change(_codec_checkpoint_update, [codec_long_run], [codec_long_checkpoint])
        long_compare_codec_run.change(_codec_checkpoint_update, [long_compare_codec_run], [long_compare_codec_checkpoint])
        diffusion_run.change(_diffusion_checkpoint_update, [diffusion_run], [compare_diffusion_checkpoint])
        diffusion_run_single.change(_diffusion_checkpoint_update, [diffusion_run_single], [diffusion_checkpoint_single])
        long_run.change(_diffusion_checkpoint_update, [long_run], [long_checkpoint])
        long_compare_diff_run.change(_diffusion_checkpoint_update, [long_compare_diff_run], [long_compare_diff_checkpoint])
        codec_run_single.change(_single_codec_genres, [codec_run_single], [codec_target])
        codec_long_run.change(_codec_longform_target, [codec_long_run], [codec_long_target])
        diffusion_run_single.change(_single_diffusion_target, [diffusion_run_single], [diffusion_target])
        long_run.change(_longform_genres, [long_run], [long_source_genre, long_target_genre])
        long_compare_diff_run.change(_longform_genres, [long_compare_diff_run], [long_compare_diff_source, long_compare_diff_target])
        long_compare_codec_run.change(_paired_compare_targets, [long_compare_codec_run, long_compare_diff_run], [long_compare_codec_target, long_compare_diff_target])
        long_compare_diff_run.change(_paired_compare_targets, [long_compare_codec_run, long_compare_diff_run], [long_compare_codec_target, long_compare_diff_target])

        compare_btn.click(
            _run_compare,
            [
                audio_input,
                example_input,
                codec_run,
                compare_codec_checkpoint,
                diffusion_run,
                compare_diffusion_checkpoint,
                compare_target,
                compare_start,
                compare_style_mode,
                compare_mix_alpha,
                compare_diffusion_seconds,
                compare_guidance,
                compare_steps,
                compare_seed,
                device_choice,
            ],
            [
                compare_md,
                compare_src_audio,
                compare_codec_audio,
                compare_diff_audio,
                compare_plot,
                compare_zip,
                compare_metrics,
                terminal_box,
            ],
        )
        codec_btn.click(
            _run_codec,
            [
                audio_input,
                example_input,
                codec_run_single,
                codec_checkpoint_single,
                codec_target,
                codec_style_mode,
                codec_mix_alpha,
                codec_start,
                codec_seed,
                device_choice,
            ],
            [codec_md, codec_src_audio, codec_out_audio, codec_plot, codec_zip, codec_metrics, terminal_box],
        )
        diffusion_btn.click(
            _run_diffusion,
            [
                audio_input,
                example_input,
                diffusion_run_single,
                diffusion_checkpoint_single,
                diffusion_target,
                diffusion_start,
                diffusion_seconds,
                diffusion_guidance,
                diffusion_steps,
                diffusion_eta,
                diffusion_seed,
                device_choice,
            ],
            [diffusion_md, diffusion_src_audio, diffusion_out_audio, diffusion_plot, diffusion_zip, diffusion_metrics, terminal_box],
        )
        real_music_btn.click(
            _run_real_music,
            [
                audio_input,
                example_input,
                real_music_checkpoint,
                real_music_target,
                real_music_seconds,
                real_music_chunk,
                real_music_overlap,
                device_choice,
            ],
            [
                real_music_md,
                real_music_src_audio,
                real_music_out_audio,
                real_music_plot,
                real_music_zip,
                real_music_metrics,
                real_music_log,
            ],
        )
        codec_long_btn.click(
            _run_codec_longform,
            [
                audio_input,
                example_input,
                codec_long_run,
                codec_long_checkpoint,
                codec_long_target,
                codec_long_style_mode,
                codec_long_mix_alpha,
                codec_long_start,
                codec_long_seconds,
                codec_long_chunk,
                codec_long_overlap,
                codec_long_seed,
                device_choice,
            ],
            [codec_long_md, codec_long_src_audio, codec_long_out_audio, codec_long_plot, codec_long_zip, codec_long_metrics, codec_long_log, terminal_box],
        )
        long_btn.click(
            _run_longform,
            [
                audio_input,
                example_input,
                long_run,
                long_checkpoint,
                long_source_genre,
                long_target_genre,
                long_start,
                long_seconds,
                long_chunk,
                long_overlap,
                long_t_start,
                long_reanchor_every,
                long_reanchor_t,
                long_guidance,
                long_style_strength,
                long_prefix,
                long_source_prefix,
                long_source_mel,
                long_hf_source,
                long_mel_time,
                long_mel_freq,
                long_assemble,
                long_ddim_steps,
                long_seed,
                device_choice,
            ],
            [long_md, long_src_audio, long_out_audio, long_plot, long_zip, long_metrics, long_log, terminal_box],
        )
        long_compare_btn.click(
            _run_longform_compare,
            [
                audio_input,
                example_input,
                long_compare_codec_run,
                long_compare_codec_checkpoint,
                long_compare_codec_target,
                long_compare_codec_style_mode,
                long_compare_codec_mix_alpha,
                long_compare_diff_run,
                long_compare_diff_checkpoint,
                long_compare_diff_source,
                long_compare_diff_target,
                long_compare_start,
                long_compare_seconds,
                long_compare_codec_chunk,
                long_compare_codec_overlap,
                long_compare_diff_chunk,
                long_compare_diff_overlap,
                long_compare_t_start,
                long_compare_reanchor_every,
                long_compare_reanchor_t,
                long_compare_guidance,
                long_compare_style_strength,
                long_compare_prefix,
                long_compare_source_prefix,
                long_compare_source_mel,
                long_compare_hf_source,
                long_compare_mel_time,
                long_compare_mel_freq,
                long_compare_assemble,
                long_compare_ddim_steps,
                long_compare_seed,
                device_choice,
            ],
            [
                long_compare_md,
                long_compare_src_audio,
                long_compare_codec_audio,
                long_compare_diff_audio,
                long_compare_codec_plot,
                long_compare_diff_plot,
                long_compare_codec_zip,
                long_compare_diff_zip,
                long_compare_metrics,
                long_compare_log,
                terminal_box,
            ],
        )
        refresh_btn.click(
            _refresh_catalog,
            [device_choice],
            [
                codec_run,
                codec_run_single,
                codec_long_run,
                long_compare_codec_run,
                compare_codec_checkpoint,
                codec_checkpoint_single,
                codec_long_checkpoint,
                long_compare_codec_checkpoint,
                diffusion_run,
                diffusion_run_single,
                long_run,
                long_compare_diff_run,
                compare_diffusion_checkpoint,
                diffusion_checkpoint_single,
                long_checkpoint,
                long_compare_diff_checkpoint,
                example_input,
                source_status,
                compare_target,
                codec_target,
                diffusion_target,
                codec_long_target,
                long_source_genre,
                long_target_genre,
                long_compare_codec_target,
                long_compare_diff_source,
                long_compare_diff_target,
                system_md,
                codec_table,
                diffusion_table,
                terminal_box,
            ],
        )
        clear_cache_btn.click(_clear_model_cache, outputs=[system_md, terminal_box])
        terminal_timer.tick(_terminal_text, outputs=[terminal_box])

    return demo
