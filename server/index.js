import express from "express";
import Anthropic from "@anthropic-ai/sdk";
import YahooFinance from "yahoo-finance2";

const yahooFinance = new YahooFinance();
const app = express();
app.use(express.json());

const client = new Anthropic();

// ---------- Price fetching ----------
async function fetchAllQuotes(tickers) {
  const quotes = {};
  await Promise.all(
    tickers.map(async (ticker) => {
      try {
        const q = await yahooFinance.quote(ticker, {}, { validateResult: false });
        if (!q || !q.regularMarketPrice) { console.warn(`No data for ${ticker}`); return; }
        quotes[ticker] = {
          name: q.shortName || q.longName || ticker,
          price: q.regularMarketPrice,
          change: q.regularMarketChange ?? null,
          changePct: q.regularMarketChangePercent ?? null,
          currency: q.currency || "USD",
        };
      } catch (err) {
        console.warn(`Failed to fetch ${ticker}:`, err.message);
      }
    })
  );
  console.log("Quotes received:", Object.keys(quotes).join(", ") || "none");
  return quotes;
}

// ---------- RSS news fetching ----------
// We pull today's headlines from two feeds:
//   E24.no  — Norwegian business news, directly covers Oslo Børs stocks
//   Reuters Markets — global macro context
// RSS is free, no API key, and gives Claude real events to anchor the brief.
const RSS_FEEDS = [
  { name: "E24",    url: "https://e24.no/rss/feed" },
  { name: "Reuters Markets", url: "https://feeds.reuters.com/reuters/businessNews" },
];

async function fetchHeadlines() {
  const headlines = [];
  const today = new Date().toDateString();

  await Promise.all(
    RSS_FEEDS.map(async (feed) => {
      try {
        const res = await fetch(feed.url, { headers: { "User-Agent": "Mozilla/5.0" } });
        if (!res.ok) { console.warn(`RSS error ${feed.name}:`, res.status); return; }
        const xml = await res.text();

        // Parse <item> blocks — pull title and pubDate without a full XML parser.
        const items = xml.match(/<item>[\s\S]*?<\/item>/g) || [];
        for (const item of items.slice(0, 15)) {
          const title   = item.match(/<title><!\[CDATA\[(.*?)\]\]><\/title>/)?.[1]
                       ?? item.match(/<title>(.*?)<\/title>/)?.[1];
          const pubDate = item.match(/<pubDate>(.*?)<\/pubDate>/)?.[1];

          if (!title) continue;

          // Only include items published today.
          const pub = pubDate ? new Date(pubDate).toDateString() : today;
          if (pub !== today) continue;

          headlines.push(`[${feed.name}] ${title.trim()}`);
        }
      } catch (err) {
        console.warn(`RSS fetch failed for ${feed.name}:`, err.message);
      }
    })
  );

  console.log(`Headlines fetched: ${headlines.length}`);
  return headlines;
}

// ---------- API route ----------
app.post("/api/market", async (req, res) => {
  const { tickers } = req.body;
  if (!Array.isArray(tickers) || !tickers.length) {
    return res.status(400).json({ error: "tickers array required" });
  }

  // Fetch prices and headlines in parallel — no point waiting for one before the other.
  const [quotes, headlines] = await Promise.all([
    fetchAllQuotes(tickers),
    fetchHeadlines(),
  ]);

  let brief = "";
  try {
    const resolved = Object.entries(quotes);
    if (resolved.length > 0) {
      const priceLines = resolved
        .map(([t, q]) => `${t} (${q.name}): ${q.price} ${q.currency}, day ${q.changePct >= 0 ? "+" : ""}${q.changePct?.toFixed(2)}%`)
        .join("\n");

      const newsSection = headlines.length > 0
        ? `\nToday's market headlines:\n${headlines.join("\n")}`
        : "";

      const message = await client.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 400,
        messages: [{
          role: "user",
          content:
            `You are writing a morning brief for a Norwegian retail investor.\n\n` +
            `Portfolio & watchlist prices:\n${priceLines}` +
            `${newsSection}\n\n` +
            `Write a short paragraph in plain English covering what moved notably today and why, ` +
            `grounded in the headlines where relevant. Cover the most important moves for this specific portfolio. ` +
            `No markdown, no headers, no bullet points, no hashtags — plain prose only.`,
        }],
      });
      brief = message.content?.[0]?.text?.trim() || "";
    }
  } catch (err) {
    console.error("Brief generation failed:", err.message);
  }

  res.json({ quotes, brief });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Ledger server running on :${PORT}`));
