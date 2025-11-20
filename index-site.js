// index-site.js
import 'dotenv/config';
import fetch from 'node-fetch';
import fs from 'fs';
import { OpenAI } from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const BASE_URL = process.env.BASE_URL || 'https://www.camjmerriwether.com';
const MAX_PAGES = 30;   // safety guard
const MAX_DEPTH = 3;    // how deep the crawler goes

const visited = new Set();

function normalizeUrl(url) {
  try {
    const u = new URL(url, BASE_URL);
    if (u.origin !== new URL(BASE_URL).origin) return null;
    // strip hash/query
    u.hash = '';
    u.search = '';
    return u.toString();
  } catch {
    return null;
  }
}

async function fetchHtml(url) {
  console.log(`Fetching ${url}`);
  const res = await fetch(url);
  if (!res.ok) {
    console.warn(`Failed ${url}: ${res.status}`);
    return null;
  }
  return await res.text();
}

// extremely basic HTML → text stripper
function htmlToText(html) {
  // remove scripts/styles
  html = html.replace(/<script[\s\S]*?<\/script>/gi, '');
  html = html.replace(/<style[\s\S]*?<\/style>/gi, '');
  // strip tags
  let text = html.replace(/<[^>]+>/g, ' ');
  // collapse whitespace
  text = text.replace(/\s+/g, ' ').trim();
  return text;
}

function extractLinks(html, baseUrl) {
  const regex = /href=["']([^"']+)["']/gi;
  const links = [];
  let match;
  while ((match = regex.exec(html)) !== null) {
    const norm = normalizeUrl(match[1]);
    if (norm && norm.startsWith(baseUrl)) links.push(norm);
  }
  return links;
}

// simple chunker to keep context sizes manageable
function chunkText(text, maxChars = 1500) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + maxChars, text.length);
    chunks.push(text.slice(start, end));
    start = end;
  }
  return chunks;
}

async function crawl(url, depth = 0, results = []) {
  if (results.length >= MAX_PAGES) return results;
  const norm = normalizeUrl(url);
  if (!norm || visited.has(norm) || depth > MAX_DEPTH) return results;

  visited.add(norm);

  const html = await fetchHtml(norm);
  if (!html) return results;

  const text = htmlToText(html);
  if (text.length > 200) {
    results.push({ url: norm, text });
  }

  const links = extractLinks(html, new URL(BASE_URL).origin);
  for (const link of links) {
    if (results.length >= MAX_PAGES) break;
    await crawl(link, depth + 1, results);
  }

  return results;
}

async function buildEmbeddings() {
  console.log('Crawling site...');
  const pages = await crawl(BASE_URL);
  console.log(`Crawled ${pages.length} pages`);

  const docs = [];
  for (const page of pages) {
    const chunks = chunkText(page.text);
    for (const chunk of chunks) {
      if (chunk.length < 100) continue; // skip tiny bits
      docs.push({
        url: page.url,
        content: chunk
      });
    }
  }
  console.log(`Total chunks to embed: ${docs.length}`);

  const embeddings = [];

  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i];
    console.log(`Embedding chunk ${i + 1}/${docs.length}`);
    const res = await client.embeddings.create({
      model: 'text-embedding-3-small',
      input: doc.content
    });

    embeddings.push({
      url: doc.url,
      content: doc.content,
      embedding: res.data[0].embedding
    });
  }

  if (!fs.existsSync('data')) fs.mkdirSync('data');

  fs.writeFileSync(
    'data/portfolio-embeddings.json',
    JSON.stringify(embeddings),
    'utf-8'
  );
  console.log('Saved embeddings to data/portfolio-embeddings.json');
}

buildEmbeddings().catch((err) => {
  console.error(err);
  process.exit(1);
});