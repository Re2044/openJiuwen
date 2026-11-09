"""Runs run_agent.py while logging every hostname it resolves, to catch downloads at run time.

Usage: record_hosts.py <log file> <run_agent.py> [run_agent args...]
"""
import runpy
import socket
import sys

log_path, script = sys.argv[1], sys.argv[2]
sys.argv = [script, *sys.argv[3:]]
real_getaddrinfo = socket.getaddrinfo


def recording_getaddrinfo(host, *args, **kwargs):
    name = host.decode() if isinstance(host, bytes) else host  # HTTP clients pass the name as bytes
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(f"{name}\n")
    return real_getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = recording_getaddrinfo
runpy.run_path(script, run_name="__main__")
