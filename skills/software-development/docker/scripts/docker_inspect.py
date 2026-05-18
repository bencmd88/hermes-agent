#!/usr/bin/env python3
"""
docker_inspect.py — Pretty-print key fields from `docker inspect` JSON output.

Usage:
    docker inspect <container-or-image> | python3 docker_inspect.py
    docker inspect <container-or-image> | python3 docker_inspect.py --json

Options:
    --json    Emit a compact JSON summary instead of the human-readable report.
"""
import json
import sys


def _get(obj, *keys, default=""):
    """Safely traverse nested dicts/lists."""
    for key in keys:
        if obj is None:
            return default
        if isinstance(obj, list):
            try:
                obj = obj[int(key)]
            except (IndexError, ValueError, TypeError):
                return default
        else:
            obj = obj.get(key, default)
    return obj if obj is not None else default


def summarise_container(data: dict) -> dict:
    """Extract the most useful fields from a container inspect record."""
    state = data.get("State", {})
    config = data.get("Config", {})
    host_cfg = data.get("HostConfig", {})
    net_settings = data.get("NetworkSettings", {})
    mounts = data.get("Mounts", [])

    # Port bindings: {container_port/proto -> [{HostIp, HostPort}]}
    port_map = {}
    for container_port, bindings in (net_settings.get("Ports") or {}).items():
        if bindings:
            port_map[container_port] = [
                f"{b.get('HostIp', '0.0.0.0')}:{b.get('HostPort', '?')}"
                for b in bindings
            ]

    # Networks: name -> IP
    networks = {
        name: info.get("IPAddress", "")
        for name, info in (net_settings.get("Networks") or {}).items()
    }

    # Mounts: host path → container path (type)
    volume_list = [
        {
            "type": m.get("Type"),
            "source": m.get("Source", ""),
            "destination": m.get("Destination", ""),
            "mode": m.get("Mode", ""),
            "rw": m.get("RW", True),
        }
        for m in mounts
    ]

    return {
        "id": data.get("Id", "")[:12],
        "name": data.get("Name", "").lstrip("/"),
        "image": config.get("Image", ""),
        "status": state.get("Status", ""),
        "running": state.get("Running", False),
        "exit_code": state.get("ExitCode", 0),
        "started_at": state.get("StartedAt", ""),
        "finished_at": state.get("FinishedAt", ""),
        "restart_policy": _get(host_cfg, "RestartPolicy", "Name", default="no"),
        "platform": data.get("Platform", ""),
        "environment": config.get("Env") or [],
        "cmd": config.get("Cmd") or [],
        "entrypoint": config.get("Entrypoint") or [],
        "ports": port_map,
        "networks": networks,
        "volumes": volume_list,
    }


def summarise_image(data: dict) -> dict:
    """Extract the most useful fields from an image inspect record."""
    config = data.get("Config", {}) or {}
    repo_tags = data.get("RepoTags") or []
    return {
        "id": data.get("Id", "").replace("sha256:", "")[:12],
        "tags": repo_tags,
        "created": data.get("Created", ""),
        "size_mb": round((data.get("Size", 0) or 0) / 1_048_576, 1),
        "architecture": data.get("Architecture", ""),
        "os": data.get("Os", ""),
        "cmd": config.get("Cmd") or [],
        "entrypoint": config.get("Entrypoint") or [],
        "exposed_ports": list((config.get("ExposedPorts") or {}).keys()),
        "environment": config.get("Env") or [],
    }


def _is_container(data: dict) -> bool:
    """Heuristic: containers have a 'State' key, images do not."""
    return "State" in data


def format_report(summary: dict, record_type: str) -> str:
    lines = []

    def row(label: str, value) -> None:
        lines.append(f"  {label:<20} {value}")

    if record_type == "container":
        status_icon = "🟢" if summary["running"] else "🔴"
        lines.append(f"Container: {summary['name']}  ({summary['id']})")
        lines.append("")
        row("Status:", f"{status_icon} {summary['status']}")
        row("Image:", summary["image"])
        row("Platform:", summary["platform"])
        row("Restart Policy:", summary["restart_policy"])
        row("Started At:", summary["started_at"])
        if not summary["running"] and summary["finished_at"]:
            row("Finished At:", summary["finished_at"])
            row("Exit Code:", summary["exit_code"])

        if summary["ports"]:
            lines.append("")
            lines.append("  Port Bindings:")
            for container_port, hosts in summary["ports"].items():
                lines.append(f"    {container_port}  →  {', '.join(hosts)}")

        if summary["networks"]:
            lines.append("")
            lines.append("  Networks:")
            for net, ip in summary["networks"].items():
                lines.append(f"    {net}: {ip or '(no IP)'}")

        if summary["volumes"]:
            lines.append("")
            lines.append("  Mounts:")
            for v in summary["volumes"]:
                rw = "rw" if v["rw"] else "ro"
                lines.append(f"    [{v['type']}] {v['source']}  →  {v['destination']}  ({rw})")

        if summary["cmd"] or summary["entrypoint"]:
            lines.append("")
            if summary["entrypoint"]:
                lines.append(f"  Entrypoint: {' '.join(summary['entrypoint'])}")
            if summary["cmd"]:
                lines.append(f"  Cmd:        {' '.join(summary['cmd'])}")

        if summary["environment"]:
            lines.append("")
            lines.append("  Environment:")
            for env in summary["environment"]:
                key = env.split("=", 1)[0]
                # Mask values that look like secrets
                if any(s in key.upper() for s in ("SECRET", "PASSWORD", "TOKEN", "KEY", "PASS")):
                    lines.append(f"    {key}=*** (masked)")
                else:
                    lines.append(f"    {env}")

    else:  # image
        lines.append(f"Image: {', '.join(summary['tags']) or summary['id']}")
        lines.append("")
        row("ID:", summary["id"])
        row("Created:", summary["created"])
        row("Size:", f"{summary['size_mb']} MB")
        row("Architecture:", f"{summary['architecture']} / {summary['os']}")

        if summary["exposed_ports"]:
            lines.append("")
            lines.append("  Exposed Ports:")
            for p in summary["exposed_ports"]:
                lines.append(f"    {p}")

        if summary["cmd"] or summary["entrypoint"]:
            lines.append("")
            if summary["entrypoint"]:
                lines.append(f"  Entrypoint: {' '.join(summary['entrypoint'])}")
            if summary["cmd"]:
                lines.append(f"  Cmd:        {' '.join(summary['cmd'])}")

    return "\n".join(lines)


def main() -> None:
    emit_json = "--json" in sys.argv

    try:
        raw = sys.stdin.read()
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: could not parse JSON from docker inspect: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(records, list):
        records = [records]

    summaries = []
    for record in records:
        if _is_container(record):
            summary = summarise_container(record)
            record_type = "container"
        else:
            summary = summarise_image(record)
            record_type = "image"
        summaries.append((summary, record_type))

    if emit_json:
        output = [s for s, _ in summaries]
        print(json.dumps(output if len(output) > 1 else output[0], indent=2))
    else:
        for i, (summary, record_type) in enumerate(summaries):
            if i > 0:
                print("\n" + "─" * 60)
            print(format_report(summary, record_type))


if __name__ == "__main__":
    main()
