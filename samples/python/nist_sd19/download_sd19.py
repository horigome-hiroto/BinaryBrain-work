"""Fetch official NIST SD19 archive with TLS verification and ZIP CRC checks."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import urllib.request
import zipfile

URL = 'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip'

def download(root):
    root = Path(root).expanduser().resolve()
    if all((root/'by_class'/('%02x' % (48+i))).is_dir() for i in range(10)):
        return
    root.mkdir(parents=True,exist_ok=True)
    archive = root/'by_class.zip'
    if not archive.exists():
        partial = root/'by_class.zip.partial'
        print('Downloading official SD19 archive (large download): '+URL,flush=True)
        try:
            with urllib.request.urlopen(URL,timeout=60) as response, partial.open('wb') as out:
                expected = response.headers.get('Content-Length')
                size = 0
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk: break
                    out.write(chunk); size += len(chunk)
                if expected is not None and size != int(expected):
                    raise IOError('Incomplete download')
            partial.replace(archive)
        except Exception as exc:
            raise RuntimeError('SD19 download failed. Allow s3.amazonaws.com in environment network settings, or manually extract by_class and pass --sd19-root. '+str(exc)) from exc
    with zipfile.ZipFile(archive) as z:
        # Extract only digits. ZipFile.open verifies CRC while fully reading.
        for member in z.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe ZIP path')
            if len(path.parts)<3 or path.parts[0]!='by_class' or path.parts[1] not in {('%02x'%(48+i)) for i in range(10)}:
                continue
            target = root.joinpath(*path.parts)
            if member.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                continue  # Preserve manually supplied/extracted files.
            partial = target.with_suffix(target.suffix+'.partial')
            with z.open(member) as src, partial.open('wb') as out:
                while True:
                    chunk = src.read(1024*1024)
                    if not chunk: break
                    out.write(chunk)
            partial.replace(target)
    with archive.open('rb') as stream:
        checksum = hashlib.file_digest(stream,'sha256').hexdigest()
    (root/'download.json').write_text(json.dumps(dict(url=URL,sha256=checksum,
        note='Local provenance hash; not a publisher-provided checksum. TLS and ZIP CRC verified.'),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='./data/sd19/raw')
    download(p.parse_args().root)
