# Book Translator

Translates a PDF book from English to German using Google Gemini. Splits the book into chapters, translates each with context-aware chunking, and outputs individual chapter files plus a combined translation.

## Prerequisites

- [Bun](https://bun.sh) runtime
- A Google Gemini API key (set `GEMINI_API_KEY` in `translate.ts`)
- A PDF file at `./the peacemaker.pdf`

## Setup

```sh
bun install
```

## Usage

Translate all chapters:

```sh
bun run translate.ts
```

Translate a specific range of chapters:

```sh
bun run translate.ts --start 3 --end 7
```

## Output

Translations are saved to `./translations/`:

- `chapter_01.txt`, `chapter_02.txt`, ... (individual chapters)
- `combined_translation.txt` (all translated chapters joined)
