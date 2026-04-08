"""Root-level OpenEnv app wrapper."""

from hack_meta.server.app import app, main as _hack_meta_main

__all__ = ["app", "main"]


def main() -> None:
    _hack_meta_main()


if __name__ == "__main__":
    main()
