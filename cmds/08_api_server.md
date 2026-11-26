# 08 – Run the API server for other users 🖥️

## Start
1. Edit the API keys in the start script (`ZIMAGE_API_KEYS=...`). Use long random strings and give each user their own key.
2. Start it:
   - Windows (5090): `cmds\scripts\start_server.bat`
   - Ubuntu (H200): `bash cmds/scripts/start_server.sh`

Wait for `Server ready` (model load + 2 warm-up images). Run **only one server process per GPU**.

## Open the port
Windows (admin cmd):
```bat
netsh advfirewall firewall add rule name="ZImage API 8000" dir=in action=allow protocol=TCP localport=8000
```
Ubuntu (if ufw is active): `sudo ufw allow 8000/tcp`

## Test
On the server:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/stats -H "X-API-Key: change-me-key1"
```
From another machine on the LAN (it only needs `requests`):
```bash
python zimage_service/client.py --url http://<SERVER-IP>:8000 --key change-me-key1 --prompt "a lighthouse at dusk"
python zimage_service/client.py --url http://<SERVER-IP>:8000 --key change-me-key1 --mode async -n 2
```
Browser: `http://<SERVER-IP>:8000/docs` gives interactive API docs (click "Try it out").

Windows PowerShell (note `curl.exe`, not `curl`):
```powershell
curl.exe -X POST http://<SERVER-IP>:8000/v1/images/generate -H "X-API-Key: change-me-key1" -H "Content-Type: application/json" -d '{\"prompt\":\"a lighthouse at dusk\",\"response_format\":\"png\"}' --output out.png
```

## Keep it running
**Ubuntu (systemd)**: create `/etc/systemd/system/zimage.service`
```ini
[Unit]
Description=Z-Image-Turbo API
After=network.target

[Service]
User=<user>
Environment=ZIMAGE_API_KEYS=key1,key2
ExecStart=/bin/bash /opt/image_llm/cmds/scripts/start_server.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now zimage
journalctl -u zimage -f
```
**Windows**: Task Scheduler → Create Task → Trigger "At startup" → Action: `cmds\scripts\start_server.bat`.

## Settings (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `ZIMAGE_QUEUE_MAX` | 32 | queued jobs before returning 503 |
| `ZIMAGE_MAX_PER_KEY` | 4 | active jobs per API key before returning 429 |
| `ZIMAGE_SYNC_TIMEOUT` | 120 | seconds a sync request waits |
| `ZIMAGE_JOB_TTL` | 3600 | async jobs and images are deleted after this |
| `ZIMAGE_OUTPUT_DIR` | zimage_service/outputs | where async images are stored |
| `ZIMAGE_WARMUP` | 2 | warm-up images at startup |
| `ZIMAGE_COMPILE` | 0 | 1 = torch.compile |
| `ZIMAGE_ATTENTION` | (SDPA) | `flash` / `_flash_3` if flash-attn is installed |

This is plain HTTP. For use outside a trusted LAN, put Nginx with HTTPS in front of it (todo section D).
