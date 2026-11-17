# magazynflary audio

Turns [Magazyn Flary](https://github.com/magazynflary/prod) articles into audio. It parses an article's
Hugo markdown, reads it out loud in Polish with [ElevenLabs](https://elevenlabs.io/) text-to-speech and
wraps it in the magazine's piano jingle.

Layout of a recording (`compose()` in `article_reading.py`, timing and levels in `JingleMix`):

- intro: the jingle (`data/jingle/v2/fortepian.flac`, lossless master); over its decaying tail: "Magazyn Flary",
  the title, the author and "Czyta głos syntetyczny." ("Read by a synthetic voice."),
- the article, sections separated by pauses,
- outro: the jingle again, "Magazyn Flary" over its tail, ringing out to its end.

"Magazyn Flary" is a recording in another voice (`data/jingle/v2/magazyn-flary.flac`: the first 2 s of
`data/jingle/v0/magazyn-flary.mp3`, as in the jingle's REAPER project), not the TTS voice. The disclosure sentence is required by Apple Podcasts in the audio itself; the podcast feed
(`roles/audio_hosting` in the infrastructure repo) uses the same sentence in its descriptions - change both together.

Output: 44.1 kHz stereo, each voice part and the whole recording normalized to -16 LUFS with true peaks at most
-1 dBTP. The voice's rare peaks (plosives; ElevenLabs speech peaks 14-18 dB over its loudness) are brought down by a
look-ahead limiter, by at most 3 dB (`SPEECH_MAX_LIMITING_DB`), and so are the few spots where the voice over the
jingle peaks higher, by at most 1.5 dB (`MIX_MAX_LIMITING_DB`); the jingle alone is never limited. `.mp3` at a
constant 128 kbit/s (Apple Podcasts and Spotify recommend at least 128; constant, because apps derive duration and
seek positions from the file size).

## Usage

Requires [uv](https://docs.astral.sh/uv/). Install dependencies:

```sh
uv sync
```

Set your ElevenLabs API key (see `.env.example`); it is read from the environment:

```sh
export ELEVENLABS_API_KEY=...
```

Read an article and save it to a file:

```sh
uv run python -m magazynflary_audio read-single-article --article path/to/article.md --output article.mp3
```

`read-single-article` options:

| Option                | Description                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `--article PATH`      | Article markdown file. Defaults to `data/article/example-1-lorem-ipsum.md`.                      |
| `--output PATH`       | Save the audio to a `.wav` or `.mp3` file.                                                       |
| `--paplay`            | Play the audio through PulseAudio/PipeWire with `paplay`.                                        |
| `--tts-cache`         | Cache TTS results in `data/cache/` so re-running on the same text does not call ElevenLabs again. |

ElevenLabs usage is paid per character, so use `--tts-cache` while iterating. The cache is keyed by the TTS
configuration (model, voice, stitching, …) as well as the text, so changing the configuration does not return stale
audio. With request stitching, audio also depends on what was read before it, so after editing a section, that
section and the ones after it are synthesized again, but the ones before it still come from the cache.

### Supported article markdown

Articles use the format of the prod repo: YAML front matter with at least `title` and `author` (an author slug),
followed by a CommonMark body. The body may contain only:

- paragraphs (bold, italics and links are read as plain text),
- `##` section headers, each followed by at least one paragraph,
- blockquotes containing paragraphs,
- `{{< pullquote >}}` … `{{< /pullquote >}}` shortcodes (the markers are dropped and the content is read as
  normal text).

Anything else (other heading levels, lists, images, other shortcodes, …) makes the parser fail with a
`ParserError` pointing at the offending line, rather than silently reading something wrong.

### Reading prod articles

Read all articles published in the prod repo within a date range (inclusive, by the front matter `date`;
without `--since-date` or `--to-date` there is no limit on that side). Drafts are included unless
`--skip-drafts` is given. It clones `magazynflary/prod` and saves one file per article, named
`YYYY-MM-DD-slug.<format>`: the publication date (front matter `date`) followed by the article file name with Polish
letters transliterated and any run of characters other than `a-z0-9` turned into `-`, e.g. the article
`samotni-wśród-ludzi.md` dated 2026-05-07 is saved to `out/2026-05-07-samotni-wsrod-ludzi.mp3`, the name format
required for publishing on the audio server:

```sh
uv run python -m magazynflary_audio read-prod-articles --since-date 2026-05-01 --to-date 2026-09-30 \
    --output-dir out --output-format mp3 --skip-drafts --tts-cache
```

An article that cannot be read (invalid front matter, e.g. a missing `date` or a `draft` that is not
`true`/`false`, one the parser does not support, an author missing from `data/authors.yaml`, or one the article reader
splits into a chunk too long for the TTS) is logged as an error and skipped. An article with invalid front matter counts
as failed even if it might be outside the date range, since its date cannot be known.
Any other error, e.g. ElevenLabs being unreachable, stops the run. An article whose output file already exists is not read again
(and nothing is overwritten); it is logged as an error and counted as a separate kind of failure. Articles that
would be saved under the same name (same date, file names equal after transliteration) are all errors: none of them is
read, since choosing one would be arbitrary. At the end it logs how many articles were read fine and how
many failed, and exits with 1 if any failed.

With `--dry-run` it only logs which articles would be read and the file names they would be saved under, without
calling ElevenLabs or writing anything. Existing output files and invalid front matter are reported as errors in a
dry run too.

### Updating authors

Author names and genders (`is_female`) are stored in `data/authors.yaml`. To add new authors from the
prod repo (it clones `magazynflary/prod` and reads `content/autorzy/*.md`):

```sh
uv run python -m magazynflary_audio.authors
```

It asks about the gender of every new author, suggesting female when the first name ends with -a.
It fails if an existing author's name has changed, so such changes need to be resolved by hand.

## Code overview

All code lives in `src/magazynflary_audio/`:

- `types.py`: data types and protocols (`ArticleParser`, `TextToSpeechSystem`, `ArticleReader`),
- `article_parsing.py`: markdown to `ParsedArticle`,
- `article_reading.py`: `ParsedArticle` to `Audio` – chunking text for TTS, the intro and outro over the jingle,
  pauses,
- `elevenlabs.py`: ElevenLabs TTS client (with retries and request stitching for consistent intonation),
- `audio.py`: helpers for reading, concatenating, resampling, overlaying audio and loudness,
- `prod_repo.py`: cloning the prod repo and finding articles by publication date,
- `audio_naming.py`: names of the published audio files (`YYYY-MM-DD-slug.mp3`),
- `authors.py`: author slug to name and gender lookup and the authors file updater,
- `cli.py`: the command line interface.

## Development

Install dependencies (including dev tools):

```sh
uv sync
```

Set up the git hooks (ruff, mypy, uv lock check):

```sh
pre-commit install
```

Run all hooks manually:

```sh
pre-commit run --all-files
```

Run tests:

```sh
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs the same checks: `uv lock --check`, ruff check, ruff format, mypy and pytest.
