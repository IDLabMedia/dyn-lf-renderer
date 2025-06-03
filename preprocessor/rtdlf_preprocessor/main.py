import os
import sys
import argparse
import numpy as np

from pathlib import Path
from typing import List, Optional

from rtdlf_preprocessor.data_loader import DataLoader

from rtdlf_preprocessor.threadpool import ThreadPool
from rtdlf_preprocessor.writers import write_cameras_info, write_metadata

from rtdlf_preprocessor.generators.fragment_generator import (
    FragmentFormat,
    FragmentGenerator,
)
from rtdlf_preprocessor.generators.vertex_generator import VertexFormat, VertexGenerator


def create_parser() -> argparse.ArgumentParser:
    ## The parser ##
    parser = argparse.ArgumentParser(
        prog="rtdlf-preprocessor",
        description="The preprocessor for the RTDLF renderer.",
        add_help=False,
    )

    ## Required arguments ##
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the input directory.",
        required=True,
    )

    ## Optional arguments ##
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )
    optional.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output directory. (default: '[INPUT]/out/')",
        required=False,
    )
    optional.add_argument(
        "-v",
        "--vertex-formats",
        nargs="*",
        type=str,
        help="List of all the vertex formats to generate. Each format can be used by the RTDLF renderer.",
        required=False,
    )
    optional.add_argument(
        "-f",
        "--fragment-formats",
        nargs="*",
        type=str,
        help="List of all the fragment formats to generate. Each format can be used by the RTDLF renderer.",
        required=False,
    )
    optional.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose informational output wile the preprocessor runs. (default: disabled)",
    )
    optional.add_argument(
        "-s",
        "--grid-spacing",
        type=float,
        default=0.01,
        help="The voxel grid spacing size. Only used for voxel downsampling (default: 0.01)",
    )
    return parser


def validate_parser_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Validate the provided arguments of the parser. Returns the arguments, in their required object forms.

    The preprocessor will exit if an invalid argument is encountered.

    :param parser: The parser to validate the arguments for.
    :type parser: argparse.ArgumentParser
    :returns: The arguments in their required object forms.
    :rtype: argparse.Namespace
    """
    args = parser.parse_args()
    args.input = validate_input_arg(args.input)
    args.output = validate_output_arg(args)

    args.vertex_formats = validate_vertex_arg(args)
    args.fragment_formats = validate_fragment_arg(args)

    return args


def validate_input_arg(input: str) -> Path:
    path = Path(input).resolve()
    if not path.is_dir():
        print(f"ERROR::INPUT::INVALID_PATH: {path}", file=sys.stderr)
        print(
            f"Please ensure the input path goes to a valid directory.", file=sys.stderr
        )
        exit(1)
    return path


def validate_output_arg(args: argparse.Namespace) -> Path:
    output: Optional[str] = args.output

    # select path
    path: Path
    if output is None:
        path = args.input / "out"
        if args.verbose:
            print(f"Using default output path: {path}")
    else:
        path = Path(output).resolve()

    if (not path.is_dir()) and args.verbose:
        print(f"Created non existing output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_vertex_arg(args: argparse.Namespace) -> List[VertexFormat]:
    # check default case
    formats_str: List[str]
    if args.vertex_formats is None:
        return []
    else:
        formats_str = args.vertex_formats

    # iterate requested formats
    formats: List[VertexFormat] = []
    for format_str in formats_str:
        # validate format and append to output
        format: VertexFormat
        try:
            format = VertexFormat(format_str.upper())
        except ValueError:
            print(f"ERROR::VERTEX::FORMAT: {format_str}", file=sys.stderr)
            print(
                f"Please ensure the vertex format is any of {[f.value for f in VertexFormat]}. (Case insensitive)",
                file=sys.stderr,
            )
            exit(1)
        formats.append(format)

    return formats


def validate_fragment_arg(args: argparse.Namespace) -> List[FragmentFormat]:
    # check default case
    formats_str: List[str]
    if args.fragment_formats is None:
        return []
    else:
        formats_str = args.fragment_formats

    # iterate requested formats
    formats: List[FragmentFormat] = []
    for format_str in formats_str:
        # validate format and append to output
        format: FragmentFormat
        try:
            format = FragmentFormat(format_str.upper())
        except ValueError:
            print(f"ERROR::FRAGMENT::FORMAT: {format_str}", file=sys.stderr)
            print(
                f"Please ensure the fragment format is any of {[f.value for f in FragmentFormat]}. (Case insensitive)",
                file=sys.stderr,
            )
            exit(1)
        formats.append(format)

    return formats


def main():
    parser = create_parser()  # read the input
    args = validate_parser_args(parser)  # validate the input

    ## run the program ##
    # load the cameras and videos
    cameras, depth_vids, color_vids = DataLoader(args.input).load_data()
    max_frames = color_vids[0].total_frames()

    # generate camera metadata file
    ThreadPool(max_workers=2).submit_task(
        write_cameras_info, out_path=args.output, cameras=cameras
    )
    print("Generating camera information... Done")
    write_metadata(
        args.output,
        color_vids[0].total_frames(),
        args.grid_spacing,
        np.array(cameras[0].model),
    )
    print("Generating metadata... Done")

    # generate vertex formats
    v_generator = VertexGenerator(
        args.output, cameras, max_frames, depth_vids, color_vids, args.grid_spacing
    )
    for v_format in args.vertex_formats:
        v_generator.generate(v_format)

    # generate fragment formats
    f_generator = FragmentGenerator(args.output, cameras, max_frames, color_vids)
    for f_format in args.fragment_formats:
        f_generator.generate(f_format)

    # close the ThreadPool
    ThreadPool().shutdown()


if __name__ == "__main__":
    main()
