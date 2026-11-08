#!/usr/bin/env bash
# Rebuilds the demo when main has new commits. Run by animerec-deploy.timer.
set -euo pipefail
cd /opt/animerec
exec 9>/run/animerec-deploy.lock
flock -n 9 || exit 0

git fetch --quiet origin main
local_rev=$(git rev-parse HEAD)
remote_rev=$(git rev-parse origin/main)
if [ "$local_rev" = "$remote_rev" ] && [ "${1:-}" != "--force" ]; then
  exit 0
fi

echo "Deploying ${remote_rev:0:7} (was ${local_rev:0:7})"
git merge --ff-only --quiet origin/main
docker compose up -d --build --remove-orphans

for _ in $(seq 1 30); do
  if curl -fsS --max-time 3 http://127.0.0.1:3003/_stcore/health >/dev/null; then
    echo "Deployed ${remote_rev:0:7}"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 2
done
echo "New version did not become healthy" >&2
exit 1
