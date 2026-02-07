"""CLI interface for Animated Talker."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="animated-talker",
        description="Generate talking-head videos with lip-sync and head motion from a portrait image and audio file.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- generate command ---
    gen = subparsers.add_parser(
        "generate",
        help="Generate a talking-head video",
    )
    gen.add_argument(
        "--image", "-i",
        required=True,
        help="Path to portrait image (JPEG/PNG)",
    )
    gen.add_argument(
        "--audio", "-a",
        required=True,
        help="Path to audio file (MP3/WAV)",
    )
    gen.add_argument(
        "--output", "-o",
        required=True,
        help="Path for output video (MP4)",
    )
    gen.add_argument(
        "--engine", "-e",
        default="sadtalker",
        choices=["sadtalker", "musetalk"],
        help="Animation engine (default: sadtalker). "
             "sadtalker = head motion + lip-sync, musetalk = lip-sync only (faster)",
    )
    gen.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video frame rate (default: 25)",
    )

    # SadTalker-specific options
    sadtalker_group = gen.add_argument_group("SadTalker options")
    sadtalker_group.add_argument(
        "--pose-style",
        type=int,
        default=0,
        help="Head motion style 0-45 (default: 0)",
    )
    sadtalker_group.add_argument(
        "--expression-scale",
        type=float,
        default=1.0,
        help="Expression intensity 0.0-3.0 (default: 1.0)",
    )
    sadtalker_group.add_argument(
        "--still-mode",
        action="store_true",
        help="Reduce background motion (useful for full-body images)",
    )
    sadtalker_group.add_argument(
        "--no-blink",
        action="store_true",
        help="Disable automatic eye blinks",
    )
    sadtalker_group.add_argument(
        "--size",
        type=int,
        default=256,
        choices=[256, 512],
        help="Face region size (default: 256)",
    )
    sadtalker_group.add_argument(
        "--preprocess",
        default="crop",
        choices=["crop", "resize", "full", "extcrop", "extfull"],
        help="Face preprocessing mode (default: crop)",
    )

    # MuseTalk-specific options
    musetalk_group = gen.add_argument_group("MuseTalk options")
    musetalk_group.add_argument(
        "--bbox-shift",
        type=int,
        default=0,
        help="Jaw boundary shift in pixels (default: 0)",
    )

    # Post-processing options
    post_group = gen.add_argument_group("post-processing")
    post_group.add_argument(
        "--face-restore",
        action="store_true",
        help="Apply CodeFormer face restoration",
    )
    post_group.add_argument(
        "--upscale",
        action="store_true",
        help="Apply Real-ESRGAN upscaling",
    )
    post_group.add_argument(
        "--upscale-factor",
        type=int,
        default=4,
        help="Upscale multiplier (default: 4)",
    )
    post_group.add_argument(
        "--restore-fidelity",
        type=float,
        default=0.7,
        help="CodeFormer fidelity 0.0-1.0 (default: 0.7)",
    )

    # Common options
    common_group = gen.add_argument_group("common")
    common_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frames per inference batch (default: 8)",
    )
    common_group.add_argument(
        "--no-float16",
        action="store_true",
        help="Disable FP16 inference (use FP32)",
    )
    common_group.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cuda, mps, cpu (default: auto)",
    )
    common_group.add_argument(
        "--model-dir",
        default=None,
        help="Custom model directory",
    )
    common_group.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Video quality 0-51, lower is better (default: 18)",
    )
    common_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # --- download-models command ---
    dl = subparsers.add_parser(
        "download-models",
        help="Download all required model weights",
    )
    dl.add_argument(
        "--engine",
        default="sadtalker",
        choices=["sadtalker", "musetalk"],
        help="Download models for this engine (default: sadtalker)",
    )
    dl.add_argument(
        "--model-dir",
        default=None,
        help="Custom model directory",
    )
    dl.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # --- check command ---
    chk = subparsers.add_parser(
        "check",
        help="Check which models are downloaded",
    )
    chk.add_argument(
        "--engine",
        default=None,
        choices=["sadtalker", "musetalk"],
        help="Check models for a specific engine (default: check all)",
    )
    chk.add_argument(
        "--model-dir",
        default=None,
        help="Custom model directory",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "generate":
        from animated_talker.pipeline import generate

        generate(
            image=args.image,
            audio=args.audio,
            output=args.output,
            engine=args.engine,
            fps=args.fps,
            # SadTalker
            pose_style=args.pose_style,
            expression_scale=args.expression_scale,
            still_mode=args.still_mode,
            use_blink=not args.no_blink,
            size=args.size,
            preprocess=args.preprocess,
            # MuseTalk
            bbox_shift=args.bbox_shift,
            # Post-processing
            face_restore=args.face_restore,
            upscale=args.upscale,
            upscale_factor=args.upscale_factor,
            restore_fidelity=args.restore_fidelity,
            # Common
            batch_size=args.batch_size,
            use_float16=not args.no_float16,
            device=args.device,
            model_dir=args.model_dir,
            crf=args.crf,
        )

    elif args.command == "download-models":
        from animated_talker.model_manager import ensure_models
        from pathlib import Path

        model_dir = Path(args.model_dir) if args.model_dir else None
        paths = ensure_models(model_dir=model_dir, engine=args.engine)
        print(f"Downloaded {len(paths)} model files for engine '{args.engine}':")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    elif args.command == "check":
        from animated_talker.model_manager import verify_models
        from pathlib import Path

        model_dir = Path(args.model_dir) if args.model_dir else None
        status = verify_models(model_dir=model_dir, engine=args.engine)
        all_ok = all(status.values())
        for key, present in status.items():
            marker = "OK" if present else "MISSING"
            print(f"  [{marker}] {key}")
        if not all_ok:
            print("\nRun 'animated-talker download-models' to download missing models.")
            sys.exit(1)
        else:
            print("\nAll models present.")


if __name__ == "__main__":
    main()
