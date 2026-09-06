/**
 * PageRenderer - Renders Quran pages using QCF4 fonts (FIXED VERSION)
 */
class PageRenderer {
    constructor(dataLoader) {
        this.dataLoader = dataLoader;
        this.container = null;
        this.loadedFonts = new Set(); // Track loaded fonts
    }

    /**
     * Initialize renderer with container element
     */
    init(containerElement) {
        this.container = containerElement;
    }

    /**
     * Render a specific page
     * @param {number} pageNumber - Page number (1-604)
     */
    async renderPage(pageNumber) {
        if (!this.container) {
            throw new Error('Renderer not initialized. Call init() first.');
        }

        // Clear container
        this.container.innerHTML = '';

        // Get layout data for this page
        const layoutData = this.dataLoader.getPageLayout(pageNumber);

        if (!layoutData || layoutData.length === 0) {
            this.container.innerHTML = '<p class="error">لا توجد بيانات لهذه الصفحة</p>';
            return;
        }

        // Get unique fonts needed for this page
        const fontsNeeded = new Set(layoutData.map(g => g.fontFile));

        // Load fonts for this page
        await Promise.all([...fontsNeeded].map(fontNum => this.loadFont(fontNum)));

        // Get metadata
        const metadata = this.dataLoader.getData().mushafMetadata;
        const centeredLines = metadata.centered_lines[pageNumber.toString()] || [];
        const headerOffset = metadata.custom_header_surah_glyph_offset_fix[pageNumber.toString()] || 0;

        // Group glyphs by line
        const lines = this.groupByLine(layoutData);

        // Create page element
        const pageEl = document.createElement('div');
        pageEl.className = 'quran-page';
        pageEl.dataset.page = pageNumber;

        // Set page face (odd = right, even = left)
        const pageFace = pageNumber % 2 === 1 ? 'right' : 'left';
        pageEl.dataset.pageFace = pageFace;

        // Add header
        const headerEl = this.renderHeader(pageNumber);
        pageEl.appendChild(headerEl);

        // Add page content container
        const contentEl = document.createElement('div');
        contentEl.className = 'page-content';

        // Render each line
        Object.keys(lines).sort((a, b) => parseInt(a) - parseInt(b)).forEach(lineNo => {
            const lineGlyphs = lines[lineNo];
            const lineEl = this.renderLine(lineGlyphs, parseInt(lineNo), centeredLines, headerOffset);
            contentEl.appendChild(lineEl);
        });

        // Mushaf Madina has exactly 15 lines per page
        // Fill remaining lines with empty lines to maintain consistent height
        const renderedLineCount = Object.keys(lines).length;
        const LINES_PER_PAGE = 15;

        for (let i = renderedLineCount; i < LINES_PER_PAGE; i++) {
            const emptyLine = document.createElement('div');
            emptyLine.className = 'quran-line empty-line';
            emptyLine.innerHTML = '&nbsp;'; // Non-breaking space to maintain height
            contentEl.appendChild(emptyLine);
        }

        pageEl.appendChild(contentEl);

        // Add footer
        const footerEl = this.renderFooter(pageNumber, pageFace);
        pageEl.appendChild(footerEl);

        this.container.appendChild(pageEl);
    }

    /**
     * Load a specific QCF4 font dynamically
     */
    async loadFont(fontNumber) {
        // Already loaded?
        if (this.loadedFonts.has(fontNumber)) {
            return;
        }

        // Create font family name
        const fontFamily = `qcf4${fontNumber}`;

        // Determine font file URL
        let fontUrl;
        if (fontNumber === 0) {
            fontUrl = 'assets/fonts/qcf4/QCF4_QBSML.woff2';
        } else {
            fontUrl = `assets/fonts/qcf4/QCF4_Hafs_${fontNumber.toString().padStart(2, '0')}_W.woff2`;
        }

        // Create and inject @font-face style
        const styleEl = document.createElement('style');
        styleEl.textContent = `
      @font-face {
        font-family: ${fontFamily};
        src: url(${fontUrl});
        font-display: block;
      }
    `;
        document.head.appendChild(styleEl);

        // Mark as loaded
        this.loadedFonts.add(fontNumber);
    }

    /**
     * Group glyphs by line number
     */
    groupByLine(layoutData) {
        const lines = {};

        layoutData.forEach(glyph => {
            const lineNo = glyph.lineNo;
            if (!lines[lineNo]) {
                lines[lineNo] = [];
            }
            lines[lineNo].push(glyph);
        });

        return lines;
    }

    /**
     * Render a single line
     */
    renderLine(glyphs, lineNo, centeredLines, headerOffset) {
        const lineEl = document.createElement('div');
        lineEl.className = 'quran-line';
        lineEl.dataset.line = lineNo;

        // Apply centering if needed
        if (centeredLines.includes(lineNo)) {
            lineEl.classList.add('centered');
        }

        // Apply header offset if line 1 and offset exists
        if (lineNo === 1 && headerOffset !== 0) {
            lineEl.style.marginTop = `${headerOffset}px`;
        }

        // Group glyphs by ayah (verse) for highlighting
        let currentAyahGroup = null;
        let currentSura = null;
        let currentVerse = null;

        glyphs.forEach(glyph => {
            // Old app logic: Only wrap in ayah-group if type is NOT 4 (basmalah) and NOT 5 (surah-name)
            // Type 4 = basmalah, Type 5 = surah-name - these should NOT be highlightable
            const shouldWrapInAyahGroup = glyph.type !== 4 && glyph.type !== 5;

            if (!shouldWrapInAyahGroup) {
                // Basmalah and surah-name - render directly without ayah grouping
                const glyphEl = this.renderGlyph(glyph);
                lineEl.appendChild(glyphEl);
                return;
            }

            // Check if we need a new ayah group
            if (currentSura !== glyph.sura || currentVerse !== glyph.verse) {
                // Close previous group if exists
                if (currentAyahGroup) {
                    lineEl.appendChild(currentAyahGroup);
                }

                // Create new ayah group
                currentAyahGroup = document.createElement('span');
                currentAyahGroup.className = 'ayah-group';
                currentAyahGroup.dataset.surah = glyph.sura;
                currentAyahGroup.dataset.ayah = glyph.verse;

                currentSura = glyph.sura;
                currentVerse = glyph.verse;
            }

            // Add glyph to current ayah group
            const glyphEl = this.renderGlyph(glyph);
            currentAyahGroup.appendChild(glyphEl);
        });

        // Append last ayah group
        if (currentAyahGroup) {
            lineEl.appendChild(currentAyahGroup);
        }

        return lineEl;
    }

    /**
     * Render a single glyph
     */
    renderGlyph(glyph) {
        const span = document.createElement('span');

        // Set QCF4 font via inline style (like original app)
        span.style.fontFamily = `qcf4${glyph.fontFile}`;

        // QCF4 fonts use Unicode Private Use Area starting at 0xF100 (61696)
        const unicodeChar = 61696 + glyph.fontCode;
        span.textContent = String.fromCharCode(unicodeChar);

        // Add data attributes
        span.dataset.type = glyph.type;
        span.dataset.sura = glyph.sura;
        span.dataset.verse = glyph.verse;
        span.dataset.fontFile = glyph.fontFile;
        span.dataset.fontCode = glyph.fontCode;

        // Add type-specific classes
        const typeClasses = {
            1: 'word',
            2: 'waqf',
            4: 'basmalah',
            5: 'surah-name',
            6: 'ayah-marker',
            7: 'rubu-marker'
        };

        // Add base qcf class for all glyphs
        span.classList.add('qcf');

        if (typeClasses[glyph.type]) {
            span.classList.add(typeClasses[glyph.type]);
        }

        return span;
    }

    /**
     * Get page info (surah, ayah range)
     */
    getPageInfo(pageNumber) {
        return this.dataLoader.getPageInfo(pageNumber);
    }

    /**
     * Render page header with surah name and juz
     */
    renderHeader(pageNumber) {
        const header = document.createElement('header');
        header.className = 'page-header';
        header.dataset.page = pageNumber;

        // Get page metadata
        const pageData = this.dataLoader.getData().pageMapping[String(pageNumber)];
        const suwarData = this.dataLoader.getData().suwar;
        const juzPages = this.dataLoader.getData().mushafMetadata.juz_pages;

        if (pageData && suwarData) {
            // Calculate Juz based on page number
            let juzNum = 1;
            for (let i = 0; i < juzPages.length; i++) {
                if (pageNumber >= juzPages[i]) {
                    juzNum = i + 1;
                } else {
                    break;
                }
            }

            // Juz (right - styled by CSS)
            const juzEl = document.createElement('div');
            juzEl.className = 'header-juz';
            juzEl.textContent = `الجُزْءُ ${this.getJuzName(juzNum)}`;

            // Collect all Surahs on this page
            const surahIds = new Set();

            // 1. Add Surah active at start of page (from page mapping)
            surahIds.add(pageData[0]);

            // 2. Add Surahs starting on this page
            suwarData.forEach((s, index) => {
                if (s.start_page === pageNumber) {
                    surahIds.add(index + 1);
                }
            });

            // 3. Format names
            const names = Array.from(surahIds)
                .sort((a, b) => a - b)
                .map(id => suwarData[id - 1].name_ar)
                .join(' - '); // Dash separator

            // Surah name (left - styled by CSS)
            const surahEl = document.createElement('div');
            surahEl.className = 'header-surah';
            surahEl.textContent = `سُورَةُ ${names}`;

            // Page face indicator (Middle)
            // Odd page (Right) = ◨
            // Even page (Left) = ◧
            const indicatorEl = document.createElement('div');
            indicatorEl.className = 'header-face-indicator';
            indicatorEl.textContent = pageNumber % 2 !== 0 ? '◨' : '◧';

            header.appendChild(juzEl);      // Right
            header.appendChild(indicatorEl); // Middle
            header.appendChild(surahEl);    // Left
        }

        return header;
    }

    /**
     * Render page footer with page number
     */
    renderFooter(pageNumber, pageFace) {
        const footer = document.createElement('footer');
        footer.className = 'page-footer';
        footer.dataset.page = pageNumber;
        footer.dataset.pageFace = pageFace;

        // Next Page Button (Right side in RTL if appended first -> Points Right for "Next" linear flow)
        const nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'footer-nav-arrow next-page-btn';
        nextBtn.innerHTML = '←'; // Right arrow for Next
        nextBtn.ariaLabel = 'الصفحة التالية';

        // Previous Page Button (Left side in RTL if appended last -> Points Left for "Back" linear flow)
        const prevBtn = document.createElement('button');
        prevBtn.type = 'button';
        prevBtn.className = 'footer-nav-arrow prev-page-btn';
        prevBtn.innerHTML = '→'; // Left arrow for Prev
        prevBtn.ariaLabel = 'الصفحة السابقة';

        const pageNumEl = document.createElement('span');
        pageNumEl.className = 'page-number';
        pageNumEl.textContent = this.arabicNumerals(pageNumber);

        // Order in RTL (Right to Left): [First Child] ... [Last Child]
        // User wanted: Right side = Prev, Left side = Next
        // "Switch the places... without change how the arrow look"

        footer.appendChild(prevBtn); // Rightmost (First Child in RTL)
        footer.appendChild(pageNumEl); // Center
        footer.appendChild(nextBtn); // Leftmost (Last Child in RTL)



        return footer;
    }

    /**
     * Convert English numerals to Arabic-Indic numerals
     */
    arabicNumerals(num) {
        const arabicDigits = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'];
        return String(num).split('').map(d => arabicDigits[parseInt(d)] || d).join('');
    }

    /**
     * Get Arabic juz name (written out in words)
     */
    getJuzName(juzNum) {
        const juzNames = [
            'الأَوَّل',      // 1
            'الثَّانِي',      // 2
            'الثَّالِث',      // 3
            'الرَّابِع',      // 4
            'الخَامِس',      // 5
            'السَّادِس',     // 6
            'السَّابِع',     // 7  
            'الثَّامِن',      // 8
            'التَّاسِع',      // 9
            'العَاشِر',      // 10
            'الحَادِيَ عَشَرَ',  // 11
            'الثَّانِيَ عَشَرَ', // 12
            'الثَّالِثَ عَشَرَ', // 13
            'الرَّابِعَ عَشَرَ', // 14
            'الخَامِسَ عَشَرَ', // 15
            'السَّادِسَ عَشَرَ', // 16
            'السَّابِعَ عَشَرَ', // 17
            'الثَّامِنَ عَشَرَ', // 18
            'التَّاسِعَ عَشَرَ', // 19
            'العِشْرُونَ',     // 20
            'الحَادِيَ وَالعِشْرُونَ',  // 21
            'الثَّانِيَ وَالعِشْرُونَ', // 22
            'الثَّالِثَ وَالعِشْرُونَ', // 23
            'الرَّابِعَ وَالعِشْرُونَ', // 24
            'الخَامِسَ وَالعِشْرُونَ', // 25
            'السَّادِسَ وَالعِشْرُونَ', // 26
            'السَّابِعَ وَالعِشْرُونَ', // 27
            'الثَّامِنَ وَالعِشْرُونَ', // 28
            'التَّاسِعَ وَالعِشْرُونَ', // 29
            'الثَّلَاثُونَ'     // 30
        ];

        return juzNames[juzNum - 1] || this.arabicNumerals(juzNum);
    }
}

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PageRenderer;
}
