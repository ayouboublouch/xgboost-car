import os
import sys
import tarfile
import urllib.request
from pathlib import Path

DEST_DIR = Path("scratch/t480")
DEST_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    ("pages.tar.gz", "https://github.com/diablo7663/T480/raw/master/data/cache/pages.tar.gz"),
    ("details.tar.gz", "https://github.com/diablo7663/T480/raw/master/data/cache/details.tar.gz"),
    ("progress.json", "https://raw.githubusercontent.com/diablo7663/T480/master/data/progress.json"),
]

for filename, url in FILES:
    dest_path = DEST_DIR / filename
    print(f"Downloading {filename} from {url}...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(100.0, blocknum * blocksize / totalsize * 100)
                sys.stdout.write(f"\r  {filename}: {percent:.1f}% ({blocknum * blocksize // 1024} KB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print(f"\nSaved {filename} ({dest_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"\nFailed to download {filename}: {e}")

# Check downloaded archives
for pack_name in ["pages.tar.gz", "details.tar.gz"]:
    pack_path = DEST_DIR / pack_name
    if pack_path.exists():
        print(f"\nExtracting {pack_name}...")
        try:
            with tarfile.open(pack_path, "r:gz") as tar:
                members = tar.getmembers()
                print(f"  Archive {pack_name} contains {len(members)} entries")
                tar.extractall(path=DEST_DIR)
            print(f"  Extracted {pack_name} successfully")
        except Exception as e:
            print(f"  Failed to extract {pack_name}: {e}")

pages_dir = DEST_DIR / "pages"
details_dir = DEST_DIR / "details"

print(f"\nExtracted pages count: {len(list(pages_dir.glob('*.json'))) if pages_dir.exists() else 0}")
print(f"Extracted details count: {len(list(details_dir.glob('*.json'))) if details_dir.exists() else 0}")
