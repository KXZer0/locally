"""Small command-line surface for the optional Odysseus companion."""

import argparse
import json

from core import odysseus


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect or start the optional Odysseus assistant stack.")
    parser.add_argument("action",
                        choices=("status", "start", "probe", "autostart"))
    # `autostart` alone reports; `autostart on|off` writes. This is the writer
    # the preference file lost when the web UI was retired -- a read with no
    # write is what made the saved value impossible to change.
    parser.add_argument("value", nargs="?", choices=("on", "off"), default=None,
                        help="For `autostart`: turn it on or off. Omit to "
                             "print the current setting and where it comes "
                             "from.")
    parser.add_argument("--port", type=int, default=odysseus.DEFAULT_PORT)
    parser.add_argument("--dir", dest="directory", default=None,
                        help="Odysseus checkout; otherwise use normal discovery")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--api-port", type=int, default=8000,
                        help="Locally API port exposed to the container")
    args = parser.parse_args(argv)

    if args.action == "autostart":
        if args.value is not None:
            if not odysseus.save_autostart(args.value == "on"):
                print(json.dumps({"ok": False,
                                  "detail": "The preference file could not be "
                                            "written."}))
                return 1
        enabled, why = odysseus.autostart_setting(directory=args.directory)
        print(json.dumps({"ok": True, "autostart": enabled, "reason": why}))
        return 0

    if args.action == "status":
        print(json.dumps(odysseus.status(args.port, args.directory,
                                         args.api_port)))
        return 0

    if args.action == "probe":
        result = odysseus.probe_api(args.api_port, args.directory, args.timeout)
        print(json.dumps(result))
        return 0 if result["reachable"] else 1

    ok, detail = odysseus.start(args.port, args.directory, args.timeout,
                                args.api_port)
    print(json.dumps({**odysseus.status(args.port, args.directory,
                                        args.api_port),
                      "detail": detail}))
    return 0 if ok else 1
