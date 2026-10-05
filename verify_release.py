from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
rows=json.loads((P/'manifests/files.json').read_text(encoding='utf8'))
for row in rows:
    f=P/row['path']
    assert f.is_file(), row['path']
    assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==row['sha256'],row['path']
assert not (P/'password.json').exists()
print('Verified',len(rows),'release files')
