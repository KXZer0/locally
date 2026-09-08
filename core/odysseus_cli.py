"""Small command-line surface for the optional Odysseus companion."""

import argparse
import json

from core import odysseus


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect or start the optional Odysseus assistant stack.")
    parser.add_argument("action", choices=("status", "start"))
    parser.add_argument("--port", type=int, default=odysseus.DEFAULT_PORT)
    parser.add_argument("--dir", dest="directory", default=None,
                        help="Odysseus checkout; otherwise use normal discovery")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args(argv)

    if args.action == "status":
        print(json.dumps(odysseus.status(args.port, args.directory)))
        return 0

    ok, detail = odysseus.start(args.port, args.directory, args.timeout)
    print(json.dumps({**odysseus.status(args.port, args.directory),
                      "detail": detail}))
    return 0 if ok else 1
