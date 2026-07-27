"""
sync_to_drive.py

Purpose
-------
Sync a local folder to/from a Google Drive folder, headlessly, using a Google
Cloud service account (not interactive OAuth -- Kaggle sessions can't complete
a browser login flow). This is the actual mechanism behind "persisting results
to Google Drive across Kaggle sessions" -- Kaggle has no native Drive mount.

One-time setup (do this once, outside Kaggle):
  1. Go to console.cloud.google.com, create (or select) a project.
  2. Enable the "Google Drive API" for that project (APIs & Services -> Library).
  3. IAM & Admin -> Service Accounts -> Create Service Account. No special
     roles needed for this use case.
  4. Open the new service account -> Keys -> Add Key -> Create new key -> JSON.
     This downloads a .json file -- treat it like a password.
  5. In your own Google Drive, create a folder (e.g. "AIF_experiments").
     Right-click -> Share -> paste the service account's email address
     (the "client_email" field in the JSON key) -> give it Editor access.
     This is the step that actually grants the service account access; a
     service account has NO access to your Drive by default.
  6. Note the folder's ID from its URL:
     https://drive.google.com/drive/folders/<THIS_IS_THE_FOLDER_ID>

Kaggle-side setup (per notebook):
  - Add the JSON key's contents as a Kaggle Secret (Add-ons -> Secrets),
    e.g. named GDRIVE_SERVICE_ACCOUNT_JSON.
  - Set DRIVE_FOLDER_ID (below, or via --folder-id) to the folder ID from
    step 6.

Usage (from within the Kaggle notebook / any Python env with the secret set)
-----
    python sync_to_drive.py upload   --local results_all --folder-id <ID>
    python sync_to_drive.py download --local results_all --folder-id <ID>

Behaviour
---------
- upload: zips --local into a single archive and uploads it to the Drive
  folder, OVERWRITING any existing file of the same name (so repeated calls
  don't pile up duplicate versions).
- download: looks for that same archive in the Drive folder; if found,
  downloads and extracts it over --local (existing local files are replaced
  by the Drive copy -- this is a restore, not a merge). If not found (e.g.
  first-ever run), does nothing and exits cleanly.

This is a SYNC operation (periodic snapshot), not a live filesystem mount --
Kaggle has no equivalent of Colab's drive.mount(). Called after each
(environment, blackout) combo via run_all_experiments.py's --post-combo-hook,
this gives you a Drive-backed checkpoint roughly every combo, so an
interrupted session loses at most the in-progress combo, not everything
since the last manual save.
"""

import argparse
import io
import json
import os
import shutil
import sys

ARCHIVE_NAME = "aif_results_all.zip"


def _get_drive_service():
    """Build an authenticated Drive API client from a service account key."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        print("Missing dependencies. Install with:\n"
              "  pip install google-api-python-client google-auth", file=sys.stderr)
        raise

    key_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    if not key_json:
        # Kaggle Secrets are exposed via UserSecretsClient, not plain env vars,
        # unless you've already copied them into the environment. Try that path
        # too, so this script works whether you wired the secret into os.environ
        # yourself or are calling this from the Kaggle-specific helper below.
        raise RuntimeError(
            "GDRIVE_SERVICE_ACCOUNT_JSON is not set in the environment. "
            "Load your Kaggle Secret and set it, e.g.:\n"
            "  from kaggle_secrets import UserSecretsClient\n"
            "  import os\n"
            "  os.environ['GDRIVE_SERVICE_ACCOUNT_JSON'] = "
            "UserSecretsClient().get_secret('GDRIVE_SERVICE_ACCOUNT_JSON')"
        )

    info = json.loads(key_json)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def _find_file_id(service, folder_id, filename):
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    resp = service.files().list(q=query, fields="files(id, name)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def upload(local_path, folder_id):
    from googleapiclient.http import MediaFileUpload

    if not os.path.isdir(local_path):
        print(f"Nothing to upload -- {local_path} does not exist or is not a directory.")
        return

    service = _get_drive_service()

    archive_base = os.path.join(os.path.dirname(local_path) or ".", "_sync_archive")
    archive_path = shutil.make_archive(archive_base, "zip", local_path)
    print(f"Zipped {local_path} -> {archive_path} ({os.path.getsize(archive_path)} bytes)")

    media = MediaFileUpload(archive_path, mimetype="application/zip", resumable=True)
    existing_id = _find_file_id(service, folder_id, ARCHIVE_NAME)

    if existing_id:
        service.files().update(fileId=existing_id, media_body=media).execute()
        print(f"Updated existing Drive file {ARCHIVE_NAME} (id={existing_id}).")
    else:
        metadata = {"name": ARCHIVE_NAME, "parents": [folder_id]}
        created = service.files().create(body=metadata, media_body=media, fields="id").execute()
        print(f"Created new Drive file {ARCHIVE_NAME} (id={created.get('id')}).")

    os.remove(archive_path)


def download(local_path, folder_id):
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_drive_service()
    file_id = _find_file_id(service, folder_id, ARCHIVE_NAME)

    if not file_id:
        print(f"No {ARCHIVE_NAME} found in Drive folder {folder_id} -- "
              f"nothing to restore (this is expected on a first run).")
        return

    request = service.files().get_media(fileId=file_id)
    archive_base = os.path.join(os.path.dirname(local_path) or ".", "_sync_archive")
    archive_path = archive_base + ".zip"

    with open(archive_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download progress: {int(status.progress() * 100)}%")

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)
    shutil.unpack_archive(archive_path, local_path, "zip")
    os.remove(archive_path)
    print(f"Restored {ARCHIVE_NAME} from Drive -> {local_path}")


def main():
    parser = argparse.ArgumentParser(description="Sync a local folder to/from Google Drive via a service account.")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--local", required=True, help="Local folder to sync (e.g. results_all).")
    parser.add_argument("--folder-id", required=True, help="Google Drive folder ID to sync with.")
    args = parser.parse_args()

    if args.action == "upload":
        upload(args.local, args.folder_id)
    else:
        download(args.local, args.folder_id)


if __name__ == "__main__":
    main()
