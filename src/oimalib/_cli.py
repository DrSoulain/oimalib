import argparse
from importlib.metadata import version


def main() -> None:
    """Main entry point for the OIMALIB command-line interface."""
    parser = argparse.ArgumentParser(
        prog="oimalib",
        description="OIMALIB - Optical Interferometry Modeling and Analysis CLI",
    )

    parser.add_argument("--version", action="store_true", help="Show the package version")
    parser.add_argument("--info", action="store_true", help="Display general information")

    args = parser.parse_args()

    if args.version:
        print("OIMALIB", version("oimalib"))
    elif args.info:
        print("Demo CLI - More features to be added soon!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
