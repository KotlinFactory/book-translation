import { GoogleGenerativeAI } from "@google/generative-ai";
import { extractText } from "unpdf";
import { readFile, writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";

// Configuration
const GEMINI_API_KEY = process.env.GEMINI_API_KEY!;
const PDF_PATH = "./the peacemaker.pdf";
const OUTPUT_DIR = "./translations";
const OVERLAP_CHARS = 3_000; // Characters to overlap between chunks for context
const MAX_CHUNK_SIZE = 25_000; // Max characters per chunk
const MAX_RETRIES = 3;

// Initialize Gemini with longer timeout
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: "gemini-3-pro-preview",

  // model: "gemini-3-flash-preview",
  generationConfig: {
    
    maxOutputTokens: 65536,
  }
});

interface Chapter {
  number: number;
  title: string;
  content: string;
}

interface Chunk {
  content: string;
  isFirst: boolean;
  isLast: boolean;
}

async function extractTextFromPDF(pdfPath: string): Promise<string> {
  console.log("📖 Extracting text from PDF...");
  const dataBuffer = await readFile(pdfPath);
  const uint8Array = new Uint8Array(dataBuffer);
  const { text, totalPages } = await extractText(uint8Array);
  console.log(`   Found ${totalPages} pages`);
  return Array.isArray(text) ? text.join("\n") : text;
}

function detectChapters(text: string): Chapter[] {
  console.log("📑 Detecting chapters...");

  const chapterPattern = /CHAPTER\s+(\d+)\s+([A-Z][A-Z\s,?!']+)/g;

  const chapters: Chapter[] = [];
  let matches: { index: number; number: number; title: string }[] = [];
  let match;

  while ((match = chapterPattern.exec(text)) !== null) {
    matches.push({
      index: match.index,
      number: parseInt(match[1]),
      title: match[2]?.trim() || `Chapter ${match[1]}`,
    });
  }

  const chapterMap = new Map<number, { index: number; title: string }>();
  for (const m of matches) {
    chapterMap.set(m.number, { index: m.index, title: m.title });
  }

  const uniqueChapters = Array.from(chapterMap.entries())
    .map(([num, data]) => ({ number: num, ...data }))
    .sort((a, b) => a.index - b.index);

  console.log(`   Found ${uniqueChapters.length} unique chapter markers`);

  const introMatch = text.match(/\nINTRODUCTION\n/);
  const epilogueMatch = text.match(/\nEPILOGUE\n/);

  const allSections: { number: number; title: string; index: number }[] = [];

  if (introMatch && introMatch.index) {
    allSections.push({ number: 0, title: "INTRODUCTION", index: introMatch.index });
  }

  for (const ch of uniqueChapters) {
    allSections.push(ch);
  }

  if (epilogueMatch && epilogueMatch.index) {
    allSections.push({ number: 99, title: "EPILOGUE", index: epilogueMatch.index });
  }

  allSections.sort((a, b) => a.index - b.index);

  for (let i = 0; i < allSections.length; i++) {
    const start = allSections[i].index;
    const end = i < allSections.length - 1 ? allSections[i + 1].index : text.length;
    let content = text.slice(start, end).trim();

    const stopMarkers = ["ACKNOWLEDGMENTS", "SOURCES AND BIBLIOGRAPHY", "NOTES\n"];
    for (const marker of stopMarkers) {
      const markerIdx = content.indexOf(marker);
      if (markerIdx > 0) {
        content = content.slice(0, markerIdx).trim();
      }
    }

    if (content.length > 100) {
      chapters.push({
        number: allSections[i].number,
        title: allSections[i].title,
        content,
      });
    }
  }

  if (chapters.length === 0) {
    console.log("   No chapter markers found, splitting by size...");
    return splitBySize(text);
  }

  console.log(`   Extracted ${chapters.length} chapters with content`);
  return chapters;
}

function splitBySize(text: string, maxChars: number = 15000): Chapter[] {
  const paragraphs = text.split(/\n\n+/);
  const chapters: Chapter[] = [];
  let currentContent = "";
  let chapterNum = 1;

  for (const para of paragraphs) {
    if (currentContent.length + para.length > maxChars && currentContent.length > 0) {
      chapters.push({
        number: chapterNum,
        title: `Section ${chapterNum}`,
        content: currentContent.trim(),
      });
      chapterNum++;
      currentContent = "";
    }
    currentContent += para + "\n\n";
  }

  if (currentContent.trim()) {
    chapters.push({
      number: chapterNum,
      title: `Section ${chapterNum}`,
      content: currentContent.trim(),
    });
  }

  return chapters;
}

// Find the best split point near a target position (at sentence boundary)
function findSentenceBoundary(text: string, targetPos: number, searchRange: number = 500): number {
  // Search backwards from target position for sentence endings
  const searchStart = Math.max(0, targetPos - searchRange);
  const searchEnd = Math.min(text.length, targetPos + searchRange);
  const searchText = text.slice(searchStart, searchEnd);

  // Find all sentence endings in the search range
  const sentenceEndings = [];
  const sentencePattern = /[.!?]["'\u201d\u2019]?\s+(?=[A-Z\u00C0-\u00DC"])/g;
  let match;

  while ((match = sentencePattern.exec(searchText)) !== null) {
    const absolutePos = searchStart + match.index + match[0].length;
    sentenceEndings.push(absolutePos);
  }

  // Also check for paragraph breaks
  const paragraphPattern = /\n\n+/g;
  while ((match = paragraphPattern.exec(searchText)) !== null) {
    const absolutePos = searchStart + match.index + match[0].length;
    sentenceEndings.push(absolutePos);
  }

  if (sentenceEndings.length === 0) {
    // Fallback: find any whitespace near target
    const spacePos = text.lastIndexOf(' ', targetPos);
    return spacePos > targetPos - 100 ? spacePos + 1 : targetPos;
  }

  // Find the closest sentence boundary to target that's before or at target
  let bestPos = sentenceEndings[0];
  for (const pos of sentenceEndings) {
    if (pos <= targetPos && pos > bestPos) {
      bestPos = pos;
    }
  }

  // If all boundaries are after target, use the first one
  if (bestPos > targetPos) {
    bestPos = sentenceEndings.reduce((min, pos) => pos < min ? pos : min, sentenceEndings[0]);
  }

  return bestPos;
}

// Get overlap text from the end of a chunk (last N characters, at sentence boundary)
function getOverlapText(text: string, overlapSize: number = OVERLAP_CHARS): string {
  if (text.length <= overlapSize) {
    return text;
  }

  const startPos = text.length - overlapSize;
  // Find a good starting point (sentence beginning)
  const adjustedStart = findSentenceBoundary(text, startPos, 300);

  return text.slice(adjustedStart);
}

// Split a chapter into smaller chunks for the API (smart sentence-aware splitting)
function chunkContent(content: string, maxSize: number = MAX_CHUNK_SIZE): Chunk[] {
  if (content.length <= maxSize) {
    return [{ content, isFirst: true, isLast: true }];
  }

  const chunks: Chunk[] = [];
  let position = 0;
  let chunkIndex = 0;

  while (position < content.length) {
    const remainingLength = content.length - position;

    if (remainingLength <= maxSize) {
      // Last chunk - take everything remaining
      chunks.push({
        content: content.slice(position).trim(),
        isFirst: chunkIndex === 0,
        isLast: true,
      });
      break;
    }

    // Find a good split point near maxSize
    const targetEnd = position + maxSize;
    const splitPoint = findSentenceBoundary(content, targetEnd, 1000);

    const chunkText = content.slice(position, splitPoint).trim();

    chunks.push({
      content: chunkText,
      isFirst: chunkIndex === 0,
      isLast: false,
    });

    // Move position forward, but keep overlap
    const overlap = getOverlapText(chunkText, OVERLAP_CHARS);
    position = splitPoint - overlap.length;

    // Make sure we're making progress
    if (position <= chunks[chunks.length - 1]?.content.length - maxSize) {
      position = splitPoint;
    }

    chunkIndex++;
  }

  console.log(`   📊 Chunk sizes: ${chunks.map(c => c.content.length).join(', ')} chars`);
  return chunks;
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function translateWithRetry(
  prompt: string,
  retries: number = MAX_RETRIES
): Promise<string> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const result = await model.generateContent(prompt);
      const response = result.response;
      return response.text();
    } catch (error: any) {
      console.log(`   ⚠️  Attempt ${attempt}/${retries} failed: ${error.message.slice(0, 50)}...`);

      if (attempt < retries) {
        const waitTime = attempt * 5000; // Exponential backoff
        console.log(`   ⏳ Waiting ${waitTime / 1000}s before retry...`);
        await sleep(waitTime);
      } else {
        throw error;
      }
    }
  }
  throw new Error("All retries failed");
}

async function translateChunk(
  chunk: Chunk,
  chapterTitle: string,
  previousContext: string
): Promise<string> {
  const prompt = `SYSTEM PROMPT – Literary Translation DE (Political Narrative Non-Fiction)

You are a senior literary translator and editor for political narrative non-fiction and historical biographies.

Translate the following English text into publication-ready German for a demanding readership (quality level: Spiegel/Feuilleton). The German must read like an original, not like a translation.

NON-NEGOTIABLES (highest priority)

Meaning & nuance first: Preserve intent, implications, irony, and subtext. Never "improve" facts.

Native German prose: Rebuild sentences for natural German rhythm. No English skeleton, no calques.

Formatting integrity: Keep paragraph breaks and all formatting (headings, lists, quotes, citations, emphasis). Use Markdown correctly.

Proper names & historical artifacts:

Keep proper nouns unless there is a widely established German equivalent.

Campaign slogans, ad titles, book titles, operation names, and named media artifacts stay in the original language and appear in German quotes if used as a reference (e.g., \u201eMorning in America\u201c). Only translate if a standard German title is clearly established.

"East of California" is a proper name and must NOT be translated as "östlich von Kalifornien". Keep it as "Im Osten Kaliforniens".

STYLE TARGET

Sophisticated but readable; intellectually sharp, narratively gripping.

Avoid bureaucratic German, academic stiffness, and "translator German".

Maintain atmosphere and momentum where the source does.

TRANSLATION RULES (quality gates)

A) Syntax & Rhythm

Do not mirror English word order. Reconstruct for idiomatic German.

Prefer crisp sentences over comma-heavy "snake sentences".

Avoid over-nominalization; use strong verbs.

Vary sentence length deliberately for narrative flow.

Use natural German punctuation and cadence.

B) Idiom & Metaphor Control

Never translate idioms literally. Use German equivalents or rephrase.

Eliminate mechanical calques ("prepare the ground" ≠ \u201Eden Boden bereiten\u201c in political narrative context).

Choose metaphors that fit German political journalism, not literal imagery.

C) Political & Historical Precision

Use correct German institutional terminology and context-dependent meanings (e.g., "the House" = Weißes Haus vs. Repräsentantenhaus).

Use standard German terms for procedures (Amtseinführung, Vereidigung, Kabinett, Fraktionsführung, etc.).

Render dates, offices, and titles in idiomatic German conventions.

D) Voice & Register

Preserve the narrator's stance (analytical vs. dramatic, ironic vs. earnest).

Avoid melodrama if the source is restrained, and avoid dryness if the source is vivid.

E) Consistency

Keep consistent translations for recurring institutions, programs, and terms across the text.

Keep consistent spelling of names, places, and ranks.

HEADLINES & SUBHEADS (special handling)

Headlines must be tight, punchy, and idiomatic, often shorter than the English.

Preserve rhetorical devices (contrast, parallelism, questions).

If a headline contains a named slogan/title, keep it as-is in German quotes.

Use Markdown syntax for structure: # for chapter titles, ## for subheadings, etc.

OUTPUT REQUIREMENTS

Output only the German translation.

Do not add translator notes unless explicitly requested.

Do not summarize or comment on your choices.

REFERENCE EXAMPLES (what "bad" looks like vs. what to do instead)

1) Named slogan / campaign ad title (must stay original, in quotes)

Source headline: Morning in America, twilight in the Cold War

BAD (wrong, misses artifact status):
EIN NEUER MORGEN IN AMERIKA, DÄMMERUNG IM KALTEN KRIEG?
("new" added, slogan translated, awkward tone)

GOOD (keeps the slogan as a named artifact, German headline rhythm):
\u201EMorning in America\u201c – Dämmerung im Kalten Krieg?

2) Calque / mechanical metaphor (avoid literal transfer)

BAD (literal and unnatural in German political prose):
\u201EUm den Boden für seinen Europa-Aufenthalt zu bereiten…\u201c

GOOD (idiomatic, journalistic):
\u201EZur Vorbereitung seines Europabesuchs…\u201c

3) Idiom handling (avoid word-for-word)

English: "the worst-kept secret in politics"

BAD (literal):
\u201Edas am schlechtesten gehütete Geheimnis der Politik\u201c

GOOD (idiomatic, tone-aware):
\u201Eein offenes Geheimnis der Politik\u201c
or (more narrative)
\u201Eein Geheimnis, das in Wahrheit längst keines mehr war\u201c

${previousContext ? `
CONTEXT FROM PREVIOUS SECTION (for continuity - DO NOT include in your translation):
"""
${previousContext.slice(-1500)}
"""
` : ""}
${!chunk.isFirst ? "\n[This is a continuation of the chapter - continue translating seamlessly]\n" : ""}

TEXT TO TRANSLATE:
"""
${chunk.content}
"""
`;

  return translateWithRetry(prompt);
}

async function translateChapter(
  chapter: Chapter,
  previousTranslation: string
): Promise<string> {
  const chunks = chunkContent(chapter.content);

  if (chunks.length > 1) {
    console.log(`   📦 Split into ${chunks.length} chunks for translation`);
  }

  const translatedParts: string[] = [];
  let context = previousTranslation;

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    if (chunks.length > 1) {
      console.log(`   🔄 Translating chunk ${i + 1}/${chunks.length}...`);
    }

    const translation = await translateChunk(chunk, chapter.title, context);
    translatedParts.push(translation);
    context = translation;

    // Delay between chunks to avoid rate limiting
    if (i < chunks.length - 1) {
      await sleep(2000);
    }
  }

  // Combine chunks intelligently by finding overlap in translations
  let fullTranslation = translatedParts[0];

  for (let i = 1; i < translatedParts.length; i++) {
    const newPart = translatedParts[i];

    // Try to find where the overlap ends by looking for matching text
    // Take last ~500 chars of previous translation and find it in new part
    const overlapSearch = fullTranslation.slice(-800);
    const sentences = overlapSearch.split(/(?<=[.!?])\s+/);
    const lastSentences = sentences.slice(-3).join(' ');

    // Find best match point in new translation
    let skipTo = 0;
    const newSentences = newPart.split(/(?<=[.!?])\s+/);

    for (let j = 0; j < Math.min(10, newSentences.length); j++) {
      // Check if this sentence or nearby text appears in the overlap
      const testSentence = newSentences[j];
      if (testSentence.length > 30 && lastSentences.includes(testSentence.slice(0, 40))) {
        // Found overlap, skip past it
        skipTo = newPart.indexOf(newSentences[j + 1] || '') || 0;
        break;
      }
    }

    // If no clear overlap found, estimate based on character count
    if (skipTo === 0) {
      // Skip roughly the overlap amount (translation might be similar length)
      const estimatedOverlapChars = Math.min(OVERLAP_CHARS * 1.2, newPart.length * 0.3);
      const searchPoint = Math.floor(estimatedOverlapChars);
      // Find next paragraph break after estimated overlap
      const nextPara = newPart.indexOf('\n\n', searchPoint);
      skipTo = nextPara > 0 ? nextPara + 2 : searchPoint;
    }

    const cleanPart = newPart.slice(skipTo).trim();
    if (cleanPart.length > 0) {
      fullTranslation += "\n\n" + cleanPart;
    }
  }

  return fullTranslation;
}

async function translateBook(startChapter: number = 1, endChapter?: number) {
  if (!existsSync(OUTPUT_DIR)) {
    await mkdir(OUTPUT_DIR, { recursive: true });
  }

  const text = await extractTextFromPDF(PDF_PATH);
  const chapters = detectChapters(text);

  const end = endChapter ?? chapters.length;
  console.log(`\n🔄 Translating chapters ${startChapter} to ${end} of ${chapters.length}\n`);

  let previousTranslation = "";
  const translations: string[] = [];

  for (let i = startChapter - 1; i < end && i < chapters.length; i++) {
    const chapter = chapters[i];

    console.log(`📝 Translating: ${chapter.title || `Chapter ${chapter.number}`}`);
    console.log(`   Content length: ${chapter.content.length} characters`);

    try {
      const translation = await translateChapter(chapter, previousTranslation);
      translations.push(translation);
      previousTranslation = translation;

      const filename = `${OUTPUT_DIR}/chapter_${chapter.number.toString().padStart(2, "0")}.txt`;
      await writeFile(filename, translation, "utf-8");
      console.log(`   ✅ Saved to ${filename}\n`);

      // Delay between chapters
      await sleep(3000);
    } catch (error: any) {
      console.error(`   ❌ Failed to translate chapter: ${error.message}`);
      console.log(`   Skipping to next chapter...\n`);
    }
  }

  if (translations.length > 0) {
    const fullTranslation = translations.join("\n\n---\n\n");
    await writeFile(`${OUTPUT_DIR}/combined_translation.txt`, fullTranslation, "utf-8");
    console.log(`\n✨ Translation complete! Combined file saved to ${OUTPUT_DIR}/combined_translation.txt`);
  }

  return translations;
}

// Parse command line arguments
const args = process.argv.slice(2);
let startChapter = 1;
let endChapter: number | undefined;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--start" && args[i + 1]) {
    startChapter = parseInt(args[i + 1]);
  }
  if (args[i] === "--end" && args[i + 1]) {
    endChapter = parseInt(args[i + 1]);
  }
}

console.log("🌍 Book Translator - The Peacemaker → German (Gemini 3 Pro)\n");
console.log("Usage: bun run translate.ts [--start N] [--end N]\n");

translateBook(startChapter, endChapter).catch(console.error);
