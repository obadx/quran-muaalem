# Quran Data Format Documentation

This document describes the data files extracted from the AlKetab Quran app.

## Overview

All Quran text, layout, and metadata has been extracted into separate JSON and CSV files for easier maintenance and modification.

---

## Files

### 1. `data/suwar.json` (10 KB)

**Surah (Chapter) Metadata** - Information about all 114 surahs.

**Format:** Array of objects

```json
[
  {
    "name_ar": "الفاتحة",
    "ayah_count": 7,
    "start_page": 1
  },
  {
    "name_ar": "البقرة",
    "ayah_count": 286,
    "start_page": 2
  }
  // ... 114 surahs total
]
```

**Fields:**
- `name_ar` (string): Arabic name of the surah
- `ayah_count` (number): Total number of ayahs in the surah
- `start_page` (number): First page where this surah appears (1-604)

---

### 2. `data/page_mapping.json` (27 KB)

**Page-to-Ayah Mapping** - Maps each of 604 pages to surah and ayah ranges.

**Format:** Object with page numbers as keys

```json
{
  "1": [1, 1, 1, 7],
  "2": [2, 1, 2, 5],
  "3": [2, 6, 2, 16]
  // ... 604 pages total
}
```

**Each array contains:** `[surah, startAyah, startPage, endAyah]`

**Example:** Page 1 → `[1, 1, 1, 7]`
- Surah: 1 (Al-Fatihah)
- Start Ayah: 1
- Start Page: 1  
- End Ayah: 7

---

### 3. `data/mushaf_metadata.json` (7 KB)

**Mushaf Layout Configuration** - Special layout rules for the Quran display.

**Format:** Object with configuration arrays

```json
{
  "juz_pages": [1, 22, 42, 62, ...],
  "mushaf_pgs": [1, 8, 13, 24, ...],
  "centered_lines": {
    "1": [1, 2, 3, 4, 5, 6, 7, 8],
    "255": [2],
    "604": [4, 9, 14, 15]
  },
  "custom_header_surah_glyph_offset_fix": {
    "235": -1,
    "595": -2,
    "604": -15
  }
}
```

**Fields:**

- **`juz_pages`** (array[30]): Starting page number for each of the 30 Juz
- **`mushaf_pgs`** (array[604]): Index mapping for mushaf page layout
- **`centered_lines`** (object): Which line numbers should be centered on specific pages
  - Key: page number (string)
  - Value: array of line numbers (1-15) to center
- **`custom_header_surah_glyph_offset_fix`** (object): Y-offset adjustments for surah headers
  - Key: page number (string)
  - Value: offset in pixels (negative = move up)

---

### 4. `data/quran_layout.csv` (1.8 MB) ⭐ **Core Data**

**Complete Quran Layout** - Every single character/glyph with its QCF4 font mapping.

**Format:** CSV with 88,439 rows

```csv
Sura,Verse,PageNo,LineNo,FontFile,FontCode,Type
1,0,1,1,0,0,5
1,1,1,2,1,0,1
1,1,1,2,1,1,1
1,1,1,2,1,2,1
```

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Sura` | int | Surah number (1-114) |
| `Verse` | int | Ayah number (0 = surah name/header) |
| `PageNo` | int | Page number (1-604) |
| `LineNo` | int | Line number on page (1-15) |
| `FontFile` | int | QCF4 font file index (0-20) |
| `FontCode` | int | Character code in that font file |
| `Type` | int | Glyph type (see below) |

**Glyph Types:**

| Type | Name | Description |
|------|------|-------------|
| `1` | Word | Regular Quran word/text |
| `2` | Waqf | Pause/stop mark (وقف) |
| `4` | Basmalah | بسم الله الرحمن الرحيم |
| `5` | Surah Name | Surah header decoration |
| `6` | Ayah Marker | Ayah number circle |
| `7` | Rubu Marker | Quarter (ربع) marker |

**Font File Mapping:**
- `0` = Basmalah font (`QCF4_QBSML.woff2`)
- `1-20` = Regular fonts (`QCF4_Hafs_01_W.woff2` through `QCF4_Hafs_20_W.woff2`)

**Example Row:**
```csv
2,255,13,2,1,1500,1
```
- Surah 2 (Al-Baqarah)
- Ayah 255 (Ayat al-Kursi)
- Page 13
- Line 2
- Font file 1
- Character code 1500
- Type 1 (word)

---

## Loading the Data

Use the provided `data-loader.js` module:

```javascript
// Load all data
const data = await QuranDataLoader.loadAll();

// Access data
console.log(data.suwar[0]); // First surah info
console.log(data.pageMapping['1']); // Page 1 info
console.log(data.layoutData); // All layout data

// Get specific page layout
const page1 = QuranDataLoader.getPageLayout(1);

// Get surah info
const fatihah = QuranDataLoader.getSurah(0);

// Get page info
const pageInfo = QuranDataLoader.getPageInfo(255);
```

---

## Statistics

- **Total Pages:** 604
- **Total Surahs:** 114
- **Total Juz:** 30
- **Total Layout Entries:** 88,439
- **Total Data Size:** ~1.86 MB (compressed)

---

## Notes

1. **Page numbers** are 1-indexed (1-604)
2. **Surah numbers** are 1-indexed (1-114)
3. **Ayah numbers** are 0-indexed for headers, 1-indexed for verses
4. **Line numbers** are 1-indexed (1-15 lines per page)
5. All Arabic text is rendered using **QCF4 fonts**, not Unicode text
6. The CSV contains **every glyph** needed to render the entire Quran exactly as it appears in printed Mushaf

---

## Backward Compatibility

The data loader provides aliases for old variable names:
- `as` → `suwar`
- `os` → `pageMapping`  
- `ss` → `mushafMetadata`
- `fs` → `layoutData` (parsed from CSV)
