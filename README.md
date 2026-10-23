# slycue

Mixed In Key–style hot cues for a Serato library, written into the files.

For every track that has **no hot cues yet**, slycue takes the beatgrid Serato already wrote, scores each 16-beat phrase for energy, and drops up to 8 cues where the energy changes. Cue 1 lands on the first downbeat, every cue sits on the phrase grid, and they're named `Energy N` in Mixed In Key's colours. Missing keys get detected (Camelot, like MIK) along with `EnergyLevel` and an `8A - Energy 6` comment.

Cues go into the `Serato Markers2` tag, so **Serato, DJ5ly and anything that reads Serato tags** sees them. Existing cues, keys and comments are never touched, and every change is logged so it can be undone.

```sh
# run straight from GitHub, nothing to install beyond python3 + librosa/mutagen
curl -fsSL https://raw.githubusercontent.com/AssiamahS/slycue/main/slycue.py -o /tmp/slycue.py
python3 /tmp/slycue.py scan ~/Music                 # how many tracks have cues
python3 /tmp/slycue.py run ~/Music/yt-dlp --dry-run --limit 5
python3 /tmp/slycue.py run ~/Music/yt-dlp           # quit Serato first
python3 /tmp/slycue.py undo                         # put everything back
```

The undo log lives at `~/.slycue/undo.jsonl`.
