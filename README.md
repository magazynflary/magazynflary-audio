# magazynflary audio

Turns [Magazyn Flary](https://github.com/magazynflary/prod) articles into audio. It parses an article's
Hugo markdown, reads it out loud in Polish with [ElevenLabs](https://elevenlabs.io/) text-to-speech and
wraps it in the magazine's jingles.

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

ElevenLabs usage is paid per character, so use `--tts-cache` while iterating.

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
`--skip-drafts` is given. It clones `magazynflary/prod` and saves one file per article, named after the
article file, e.g. `out/samotni-wśród-ludzi.mp3`:

```sh
uv run python -m magazynflary_audio read-prod-articles --since-date 2026-05-01 --to-date 2026-09-30 \
    --output-dir out --output-format mp3 --skip-drafts --tts-cache
```

An article that cannot be read (one the parser does not support, an author missing from
`data/authors.yaml`, or one the article reader splits into a chunk too long for the TTS) is logged as an error and skipped.
Any other error, e.g. ElevenLabs being unreachable, stops the run. An article whose output file already exists is not read again
(and nothing is overwritten); it is logged as an error and counted as a separate kind of failure. At the end it logs how many articles were read fine and how
many failed, and exits with 1 if any failed.

With `--dry-run` it only logs which articles would be read and which files they would be saved to, without
calling ElevenLabs or writing anything. Existing output files are reported as errors in a dry run too.

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
- `article_reading.py`: `ParsedArticle` to `Audio` – chunking text for TTS, adding jingles and pauses,
- `elevenlabs.py`: ElevenLabs TTS client (with retries and request stitching for consistent intonation),
- `audio.py`: helpers for concatenating, resampling and reading audio,
- `prod_repo.py`: cloning the prod repo and finding articles by publication date,
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
