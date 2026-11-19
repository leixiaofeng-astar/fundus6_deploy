#!/usr/bin/env python3
"""Create a private Docker env file without overwriting existing credentials."""
import os
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
path = root / '.env.docker'
try:
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    print('Keeping existing .env.docker and its API key.')
else:
    with os.fdopen(descriptor, 'w') as f:
        f.write('FUNDUS_API_KEY='+secrets.token_urlsafe(32)+'\n')
        f.write('FUNDUS_BIND_ADDRESS=127.0.0.1\nFUNDUS_PORT=5050\nFUNDUS_CPU_THREADS=2\n')
    print('Created .env.docker with owner-only permissions.')
print('Use the FUNDUS_API_KEY value in .env.docker as the web UI API key.')
