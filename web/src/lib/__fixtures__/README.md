# geometry.json

Real output of `server/serialize.py`'s `serialize_geometry()` (which wraps
`engine/board.py`'s `GEOMETRY` singleton) -- not hand-constructed. Regenerate
after any change to board geometry or `serialize_geometry`'s shape:

```bash
uv run python -c "
import json
from server.serialize import serialize_geometry
print(json.dumps(serialize_geometry()))
" > web/src/lib/__fixtures__/geometry.json
```
