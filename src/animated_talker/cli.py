"""CLI interface for Animated Talker."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="animated-talker",
        description="Generate lip-synced talking avatar videos from a portrait image and audio file.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- generate command ---
    gen = subparsers.add_parser(
        "generate",
        help="Generate a lip-synced video",
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
        "--fps",
        type=int,
        default=25,
        help="Video frame rate (default: 25)",
    )
    gen.add_argument(
        "--bbox-shift",
        type=int,
        default=0,
        help="Jaw boundary shift in pixels (default: 0)",
    )
    gen.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Frames per inference batch (default: 8)",
    )
    gen.add_argument(
        "--no-float16",
        action="store_true",
        help="Disable FP16 inference (use FP32)",
    )
    gen.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cuda, mps, cpu (default: auto)",
    )
    gen.add_argument(
        "--extra-margin",
        type=int,
        default=10,
        help="Extra margin below jaw in pixels (default: 10)",
    )
    gen.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # --- setup command ---
    setup_cmd = subparsers.add_parser(
        "setup",
        help="Full setup: clone MuseTalk repo, install dependencies, download models",
    )
    setup_cmd.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip installing MuseTalk dependencies",
    )
    setup_cmd.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # --- download-models command ---
    dl = subparsers.add_parser(
        "download-models",
        help="Clone MuseTalk repo and download all required model weights",
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
            fps=args.fps,
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size,
            use_float16=not args.no_float16,
            device=args.device,
            extra_margin=args.extra_margin,
        )

    elif args.command == "setup":
        from animated_talker.model_manager import setup_all

        install_deps = not args.no_deps
        print("Running full setup (install_deps=%s)..." % install_deps)
        musetalk_dir = setup_all(install_deps=install_deps)
        print(f"Setup complete! MuseTalk at: {musetalk_dir}")

    elif args.command == "download-models":
        from animated_talker.model_manager import setup_all

        print("Cloning MuseTalk repo and downloading models...")
        musetalk_dir = setup_all(install_deps=False)
        print(f"Done! MuseTalk at: {musetalk_dir}")

    elif args.command == "check":
        from animated_talker.model_manager import verify_setup

        status = verify_setup()
        all_ok = all(status.values())
        for key, present in status.items():
            marker = "OK" if present else "MISSING"
            print(f"  [{marker}] {key}")
        if not all_ok:
            print("\nRun 'animated-talker setup' to download missing components.")
            sys.exit(1)
        else:
            print("\nAll components present.")


if __name__ == "__main__":
    main()
