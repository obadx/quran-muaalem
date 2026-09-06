/**
 * QuranApp - Main Application Controller
 * 
 * PUBLIC API:
 * - quranApp.goToAyah(surah, ayah, options) - Navigate to and highlight any ayah
 * - quranApp.clearPersistentHighlight() - Clear persistent highlight
 * - quranApp.highlightAyah(surah, ayah) - Highlight ayah on current page
 * - quranApp.removeAllHighlights() - Remove all highlights
 * 
 * URL PARAMETERS:
 * - ?surah=2&ayah=6 - Navigate directly to Surah 2, Ayah 6
 * - ?s=2&a=6 - Short form of the above
 */
class QuranApp {
    constructor() {
        this.dataLoader = null;
        this.renderer = null;
        this.currentPage = 1;
        this.isReady = false;
        // Persistent highlight state (for external API calls)
        this.persistentHighlight = null;
        // Tafseer data
        this.tafseerAr = null;  // Arabic tafseer (التفسير الميسر)
        this.tafseerEn = null;  // English translation (Saheeh International)
        // Audio state
        this.currentAudio = null;
        this.currentPlayingAyah = null;
        this._highlightTimeout = null;
        // Quran Muaalem controllers
        this.quranAPI = null;
        this.searchController = null;
        this.reciteController = null;
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            // Show loading screen
            this.showLoading();

            // Load data
            console.log('Loading Quran data...');
            this.dataLoader = new QuranDataLoader();
            await this.dataLoader.loadAll();
            console.log('✓ Data loaded successfully');

            // Initialize renderer
            this.renderer = new PageRenderer(this.dataLoader);
            this.renderer.init(document.getElementById('pageContainer'));

            // Detect Safari and add class to document for CSS workarounds
            // Safari has bugs with ::before pseudo-elements on inline elements
            this.isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
            if (this.isSafari) {
                document.documentElement.classList.add('is-safari');
            }

            // Setup overlay navigation
            this.setupOverlay();

            // Setup navigation arrows
            this.setupNavArrows();

            // Setup postMessage listener for parent frame communication
            this.setupMessageListener();

            // Check URL parameters for ayah navigation first, then hash for page
            const urlAyah = this.getAyahFromURL();
            let initialPage = 1;

            if (urlAyah) {
                // Find the page containing this ayah
                initialPage = this.findAyahPage(urlAyah.surah, urlAyah.ayah) || 1;
                // Set persistent highlight to be applied after render
                this.persistentHighlight = { surah: urlAyah.surah, ayah: urlAyah.ayah };
            } else {
                initialPage = this.getPageFromHash() || 1;
            }

            // Render first page
            await this.renderPage(initialPage);

            // Load tafseer data (don't block rendering)
            this.loadTafseerData();

            // Setup tafseer overlay
            this.setupTafseerOverlay();

            // Initialize Quran Muaalem integration
            this.quranAPI = new QuranAPI();
            this.searchController = new SearchController(this, this.quranAPI);
            this.searchController.init();
            this.reciteController = new ReciteController(this, this.quranAPI);
            this.reciteController.init();

            // Check API health (non-blocking)
            this.quranAPI.health().then(h => {
                console.log('Muaalem API:', h.status, '| Engine:', h.engine_status);
            }).catch(() => {
                console.warn('Muaalem API not available — start the servers to enable search & recite');
            });

            // Mark as ready
            this.isReady = true;

            // Hide loading, show app
            this.hideLoading();

            // Notify parent frame that we're ready
            if (window.parent !== window) {
                window.parent.postMessage({ type: 'mushaf-ready' }, '*');
            }

            // Scale page to fit viewport (disabled - using CSS viewport units instead)
            // this.scaleToFit();
            // window.addEventListener('resize', () => this.scaleToFit());

            console.log('✅ App initialized successfully');

            // Log API usage for developers
            console.log('Public API: quranApp.goToAyah(surah, ayah) - Navigate to any ayah');
            console.log('URL params: ?surah=2&ayah=6 or ?s=2&a=6');
            console.log('Muaalem: Use Search/Recite buttons in the toolbar');
        } catch (error) {
            console.error('❌ Failed to initialize app:', error);
            this.showError(error);
        }
    }

    /**
     * Setup postMessage listener for communication with parent frame
     * This is more reliable than direct iframe access, especially on mobile Safari
     */
    setupMessageListener() {
        window.addEventListener('message', async (event) => {
            // Only handle messages from parent frame
            if (event.source !== window.parent) return;

            const { type, surah, ayah } = event.data || {};

            if (type === 'goToAyah' && surah && ayah) {
                try {
                    const success = await this.goToAyah(surah, ayah);
                    window.parent.postMessage({
                        type: 'goToAyah-result',
                        success,
                        surah,
                        ayah
                    }, '*');
                } catch (err) {
                    window.parent.postMessage({
                        type: 'goToAyah-result',
                        success: false,
                        error: err.message
                    }, '*');
                }
            } else if (type === 'clearHighlight') {
                this.clearPersistentHighlight();
            } else if (type === 'ping') {
                window.parent.postMessage({ type: 'pong', isReady: this.isReady }, '*');
            }
        });
    }

    /**
     * PUBLIC API: Navigate to a specific ayah and highlight it
     * @param {number|string} surah - Surah number (1-114)
     * @param {number|string} ayah - Ayah number
     * @param {Object} options - Options
     * @param {boolean} options.persistent - If true, highlight stays until clearPersistentHighlight() is called
     * @param {boolean} options.scroll - If true, scroll the ayah into view (default: true)
     * @returns {Promise<boolean>} - Returns true if successful, false if ayah not found
     *
     * @example
     * // Navigate to Surah Al-Baqarah (2), Ayah 255 (Ayat Al-Kursi)
     * await quranApp.goToAyah(2, 255);
     *
     * // With persistent highlight (stays until manually cleared)
     * await quranApp.goToAyah(2, 255, { persistent: true });
     *
     * // Later, clear the persistent highlight
     * quranApp.clearPersistentHighlight();
     */
    async goToAyah(surah, ayah, options = {}) {
        const surahNum = parseInt(surah);
        const ayahNum = parseInt(ayah);
        const { persistent = true, scroll = true } = options;

        // Validate inputs
        if (isNaN(surahNum) || surahNum < 1 || surahNum > 114) {
            console.error(`Invalid surah number: ${surah}. Must be 1-114.`);
            return false;
        }

        // Wait for app to be ready
        if (!this.isReady) {
            console.log('Waiting for app to initialize...');
            await new Promise(resolve => {
                const check = setInterval(() => {
                    if (this.isReady) {
                        clearInterval(check);
                        resolve();
                    }
                }, 100);
            });
        }

        // Find the page containing this ayah
        const page = this.findAyahPage(surahNum, ayahNum);
        if (!page) {
            console.error(`Ayah ${surahNum}:${ayahNum} not found in the Mushaf.`);
            return false;
        }

        // Clear any pending highlight timeout first
        if (this._highlightTimeout) {
            clearTimeout(this._highlightTimeout);
            this._highlightTimeout = null;
        }

        // Set persistent highlight if requested
        if (persistent) {
            this.persistentHighlight = { surah: surahNum, ayah: ayahNum };
        }

        // Navigate to the page if needed
        if (this.currentPage !== page) {
            // renderPage will handle the highlight after rendering
            await this.renderPage(page);
        } else {
            // Already on the page - wait for fonts then apply highlight
            const applyHighlight = async () => {
                // Wait for fonts to be ready
                if (document.fonts && document.fonts.ready) {
                    await document.fonts.ready;
                }

                this.removePersistentHighlight();
                this.applyPersistentHighlight(surahNum, ayahNum);

                // scrollIntoView disabled - causes page height changes with overflow:hidden containers
                // if (scroll) {
                //     const ayahElement = document.querySelector(
                //         `.ayah-group[data-surah="${surahNum}"][data-ayah="${ayahNum}"]`
                //     );
                //     if (ayahElement) {
                //         ayahElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                //     }
                // }
            };

            await applyHighlight();
        }

        console.log(`📍 Navigated to Surah ${surahNum}, Ayah ${ayahNum} (Page ${page})`);
        return true;
    }

    /**
     * PUBLIC API: Clear persistent highlight
     */
    clearPersistentHighlight() {
        this.persistentHighlight = null;
        this.removePersistentHighlight();
        console.log('🔄 Persistent highlight cleared');
    }

    /**
     * Find which page contains a specific ayah
     * @param {number} surah - Surah number (1-114)
     * @param {number} ayah - Ayah number
     * @returns {number|null} - Page number or null if not found
     */
    findAyahPage(surah, ayah) {
        const layoutData = this.dataLoader.layoutData;
        if (!layoutData) return null;

        // Find the first glyph entry for this ayah
        const entry = layoutData.find(row => row.sura === surah && row.verse === ayah);
        return entry ? entry.pageNo : null;
    }

    /**
     * Get ayah from URL parameters
     * Supports: ?surah=2&ayah=6 or ?s=2&a=6
     * @returns {Object|null} - {surah, ayah} or null
     */
    getAyahFromURL() {
        const params = new URLSearchParams(window.location.search);

        // Try full names first, then short forms
        const surah = params.get('surah') || params.get('s');
        const ayah = params.get('ayah') || params.get('a');

        if (surah && ayah) {
            const surahNum = parseInt(surah);
            const ayahNum = parseInt(ayah);
            if (!isNaN(surahNum) && !isNaN(ayahNum)) {
                return { surah: surahNum, ayah: ayahNum };
            }
        }
        return null;
    }

    /**
     * Scale page to fit viewport height (like SwiftUI aspectRatio .fit)
     */
    scaleToFit() {
        const page = document.querySelector('.quran-page');
        const container = document.querySelector('.page-container');
        if (!page || !container) return;

        // Reset transform to measure natural size
        page.style.transform = 'none';
        container.style.overflow = 'hidden';

        // Get measurements
        const viewportHeight = window.innerHeight;
        const pageHeight = page.offsetHeight;

        // Calculate scale to fit viewport
        if (pageHeight > viewportHeight) {
            const scale = (viewportHeight / pageHeight) * 0.98; // 98% to have small margin
            page.style.transform = `scale(${scale})`;
            page.style.transformOrigin = 'top center';
            // Adjust container height to match scaled content
            container.style.height = `${pageHeight * scale}px`;
        } else {
            page.style.transform = 'none';
            container.style.height = 'auto';
        }
    }

    /**
     * Render a specific page
     */
    async renderPage(pageNumber) {
        try {
            await this.renderer.renderPage(pageNumber);

            // Update current page
            this.currentPage = pageNumber;

            // Update navigation arrow states
            this.updateNavArrows();

            // Setup ayah highlighting (highlight entire ayah across all lines)
            this.setupAyahHighlighting();

            // Apply persistent highlight if set (yellow color)
            // Wait for fonts to load so we don't highlight invisible content
            if (this.persistentHighlight) {
                const targetSurah = this.persistentHighlight.surah;
                const targetAyah = this.persistentHighlight.ayah;
                const targetPage = pageNumber;

                // Clear any pending highlight
                if (this._highlightTimeout) {
                    clearTimeout(this._highlightTimeout);
                    this._highlightTimeout = null;
                }

                // Apply highlight helper
                const doApplyHighlight = () => {
                    // Verify highlight is still valid for this page
                    if (this.persistentHighlight &&
                        this.persistentHighlight.surah === targetSurah &&
                        this.persistentHighlight.ayah === targetAyah &&
                        this.currentPage === targetPage) {
                        this.applyPersistentHighlight(targetSurah, targetAyah);

                        // scrollIntoView disabled - causes page height changes with overflow:hidden containers
                        // The Mushaf page is fixed height and doesn't need scrolling
                        // const ayahElement = document.querySelector(
                        //     `.ayah-group[data-surah="${targetSurah}"][data-ayah="${targetAyah}"]`
                        // );
                        // if (ayahElement) {
                        //     ayahElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        // }
                    }
                };

                // Wait for fonts to load, then apply highlight
                const applyHighlightWhenReady = async () => {
                    try {
                        // Wait for all fonts to be loaded (with timeout for Safari)
                        if (document.fonts && document.fonts.ready) {
                            await Promise.race([
                                document.fonts.ready,
                                new Promise(resolve => setTimeout(resolve, 500)) // 500ms timeout fallback
                            ]);
                        }
                    } catch (e) {
                        console.warn('Font loading error, proceeding anyway:', e);
                    }

                    // Double rAF to ensure paint is complete after fonts load
                    requestAnimationFrame(() => {
                        requestAnimationFrame(() => {
                            doApplyHighlight();
                        });
                    });
                };

                applyHighlightWhenReady();
            }

            // Log page info
            const pageInfo = this.renderer.getPageInfo(pageNumber);
            if (pageInfo) {
                const [surah, startAyah, , endAyah] = pageInfo;
                const surahData = this.dataLoader.getSurah(surah - 1);
                console.log(`Page ${pageNumber}: ${surahData.name_ar} (${startAyah}-${endAyah})`);
            }
        } catch (error) {
            console.error('Failed to render page:', error);
            document.getElementById('pageContainer').innerHTML =
                `<p class="error">فشل تحميل الصفحة: ${error.message}</p>`;
        }
    }

    /**
     * Highlight an ayah (all parts across all lines)
     */
    highlightAyah(surah, ayah) {
        // Find all ayah-groups with matching surah/ayah
        const selector = `.ayah-group[data-surah="${surah}"][data-ayah="${ayah}"]`;
        document.querySelectorAll(selector).forEach(el => {
            el.classList.add('highlighted-verse');
        });
    }

    /**
     * Remove all hover highlights (gray)
     */
    removeAllHighlights() {
        document.querySelectorAll('.ayah-group.highlighted-verse').forEach(el => {
            el.classList.remove('highlighted-verse');
        });
    }

    /**
     * Apply persistent highlight (yellow) - for API/URL navigation
     */
    applyPersistentHighlight(surah, ayah) {
        const selector = `.ayah-group[data-surah="${surah}"][data-ayah="${ayah}"]`;
        const elements = document.querySelectorAll(selector);

        if (elements.length === 0) return;

        // Check if Safari (CSS ::before doesn't work reliably on Safari)
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

        elements.forEach(el => {
            // Force a reflow before adding class
            // eslint-disable-next-line no-unused-expressions
            el.offsetHeight;
            el.classList.add('persistent-highlight');

            // Safari workaround: Use background directly on element
            // Safari has bugs with CSS variables and animations on pseudo-elements
            if (isSafari) {
                const highlightColor = getComputedStyle(document.documentElement)
                    .getPropertyValue('--highlight-persistent').trim() || 'rgba(255, 235, 59, 0.5)';
                el.style.setProperty('--safari-highlight', highlightColor);
                el.classList.add('safari-highlight-fix');
            }
        });
    }

    /**
     * Remove persistent highlight (yellow)
     */
    removePersistentHighlight() {
        document.querySelectorAll('.ayah-group.persistent-highlight, .ayah-group.safari-highlight-fix').forEach(el => {
            el.classList.remove('persistent-highlight');
            el.classList.remove('safari-highlight-fix');
            el.style.removeProperty('--safari-highlight');
        });
    }

    /**
     * Setup ayah highlighting event listeners
     * When hovering over any part of an ayah, highlight ALL parts across all lines
     * When clicking, show tafseer modal
     */
    setupAyahHighlighting() {
        document.querySelectorAll('.ayah-group').forEach(ayahGroup => {
            ayahGroup.addEventListener('mouseenter', () => {
                const surah = ayahGroup.getAttribute('data-surah');
                const ayah = ayahGroup.getAttribute('data-ayah');
                this.highlightAyah(surah, ayah);
            });

            ayahGroup.addEventListener('mouseleave', () => {
                this.removeAllHighlights();
                // Persistent highlight stays (it uses a different class)
            });

            // Single click — select ayah (persistent highlight)
            ayahGroup.addEventListener('click', () => {
                const surah = parseInt(ayahGroup.getAttribute('data-surah'));
                const ayah = parseInt(ayahGroup.getAttribute('data-ayah'));

                // Toggle: if same ayah is already selected, deselect it
                if (this.persistentHighlight &&
                    this.persistentHighlight.surah === surah &&
                    this.persistentHighlight.ayah === ayah) {
                    this.clearPersistentHighlight();
                    this.removePersistentHighlight();
                } else {
                    // Clear old, apply new
                    this.removePersistentHighlight();
                    this.persistentHighlight = { surah, ayah };
                    this.applyPersistentHighlight(surah, ayah);
                }
            });

            // Double click — show tafseer
            ayahGroup.addEventListener('dblclick', () => {
                const surah = parseInt(ayahGroup.getAttribute('data-surah'));
                const ayah = parseInt(ayahGroup.getAttribute('data-ayah'));
                this.showTafseer(surah, ayah);
            });
        });
    }

    /**
     * Get page number from URL hash
     */
    getPageFromHash() {
        const hash = window.location.hash;
        const match = hash.match(/#page\/(\d+)/);
        if (match) {
            const page = parseInt(match[1]);
            if (page >= 1 && page <= 604) {
                return page;
            }
        }
        return null;
    }

    /**
     * Show loading screen
     */
    showLoading() {
        document.getElementById('loading').style.display = 'flex';
        document.getElementById('app').style.display = 'none';
    }

    /**
     * Hide loading screen
     */
    hideLoading() {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('app').style.display = 'flex';
    }

    /**
     * Show error message
     */
    showError(error) {
        document.getElementById('loading').innerHTML = `
      <div class="error-screen">
        <h2>⚠️ خطأ</h2>
        <p>${error.message}</p>
        <pre style="text-align: left; font-size: 12px;">${error.stack}</pre>
        <button onclick="location.reload()">إعادة المحاولة</button>
      </div>
    `;
    }

    /**
     * Setup overlay navigation
     */
    setupOverlay() {
        const overlay = document.getElementById('overlay');
        const navTabs = document.querySelectorAll('.nav-tab');

        // Populate lists
        this.populateSurahsList();
        this.populateJuzList();

        // Tab switching
        navTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.dataset.tab;
                navTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                document.getElementById('surahs-list').style.display = tabName === 'surahs' ? 'flex' : 'none';
                document.getElementById('juz-list').style.display = tabName === 'juz' ? 'flex' : 'none';
            });
        });

        // Close overlay on background click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.closeOverlay();
            }
        });

        // Close button
        const closeBtn = document.getElementById('overlayClose');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeOverlay());
        }

        // Add click handlers to page headers/footers
        document.addEventListener('click', (e) => {
            // Provide a way to ignore clicks on navigation arrows
            if (e.target.closest('.footer-nav-arrow')) {
                return;
            }

            if (e.target.closest('.page-header') || e.target.closest('.page-footer')) {
                this.openOverlay();
            }
        });
    }

    /**
     * Setup navigation arrows for prev/next page
     */
    /**
     * Setup navigation arrows for prev/next page
     */
    setupNavArrows() {
        // Event delegation for footer navigation buttons
        const pageContainer = document.getElementById('pageContainer');
        if (pageContainer) {
            pageContainer.addEventListener('click', (e) => {
                if (e.target.closest('.prev-page-btn')) {
                    this.prevPage();
                } else if (e.target.closest('.next-page-btn')) {
                    this.nextPage();
                }
            });
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            // Ignore if any overlay is active
            if (document.getElementById('overlay').classList.contains('active') ||
                document.getElementById('tafseerOverlay').classList.contains('active') ||
                document.getElementById('resultsOverlay').classList.contains('active')) {
                return;
            }

            if (e.key === 'ArrowRight') {
                this.prevPage(); // Right goes to previous page in RTL
            } else if (e.key === 'ArrowLeft') {
                this.nextPage(); // Left goes to next page in RTL
            } else if (e.key === ' ' || e.code === 'Space') {
                e.preventDefault(); // Prevent scrolling
                this.openOverlay(); // Space opens navigation overlay
            }
        });
    }

    /**
     * Update navigation arrow states (deprecated - specific to old buttons)
     */
    updateNavArrows() {
        // No-op for now as buttons are re-rendered with page
    }

    /**
     * Update navigation arrow states (disable at boundaries)
     */
    updateNavArrows() {
        const prevBtn = document.getElementById('prevPageBtn');
        const nextBtn = document.getElementById('nextPageBtn');
        const totalPages = 604; // Mushaf has 604 pages

        if (prevBtn) {
            prevBtn.disabled = this.currentPage <= 1;
        }
        if (nextBtn) {
            nextBtn.disabled = this.currentPage >= totalPages;
        }
    }

    /**
     * Go to previous page (clears persistent highlight)
     */
    prevPage() {
        if (this.currentPage > 1) {
            this.persistentHighlight = null;  // Clear on manual navigation
            this.renderPage(this.currentPage - 1);
        }
    }

    /**
     * Go to next page (clears persistent highlight)
     */
    nextPage() {
        const totalPages = 604;
        if (this.currentPage < totalPages) {
            this.persistentHighlight = null;  // Clear on manual navigation
            this.renderPage(this.currentPage + 1);
        }
    }

    /**
     * Populate surahs list
     */
    populateSurahsList() {
        const list = document.getElementById('surahs-list');
        const suwar = this.dataLoader.getData().suwar;

        list.innerHTML = suwar.map((surah, index) => `
            <div class="nav-list-item" data-page="${surah.start_page}">
                <span class="item-number">${this.arabicNumerals(index + 1)}</span>
                <span class="item-name">${surah.name_ar}</span>
                <span class="item-page">ص ${this.arabicNumerals(surah.start_page)}</span>
            </div>
        `).join('');

        // Add click handlers
        list.querySelectorAll('.nav-list-item').forEach(item => {
            item.addEventListener('click', () => {
                const page = parseInt(item.dataset.page);
                this.persistentHighlight = null;  // Clear on manual navigation
                this.renderPage(page);
                this.closeOverlay();
            });
        });
    }

    /**
     * Populate juz list
     */
    populateJuzList() {
        const list = document.getElementById('juz-list');
        const juzPages = this.dataLoader.getData().mushafMetadata.juz_pages;

        list.innerHTML = juzPages.slice(0, 30).map((page, index) => `
            <div class="nav-list-item" data-page="${page}">
                <span class="item-number">${this.arabicNumerals(index + 1)}</span>
                <span class="item-name">الجُزْءُ ${this.getJuzName(index + 1)}</span>
                <span class="item-page">ص ${this.arabicNumerals(page)}</span>
            </div>
        `).join('');

        // Add click handlers
        list.querySelectorAll('.nav-list-item').forEach(item => {
            item.addEventListener('click', () => {
                const page = parseInt(item.dataset.page);
                this.persistentHighlight = null;  // Clear on manual navigation
                this.renderPage(page);
                this.closeOverlay();
            });
        });
    }

    /**
     * Open overlay
     */
    openOverlay() {
        const overlay = document.getElementById('overlay');
        overlay.classList.add('active');

        // Update overlay title with current page info
        const pageNum = this.currentPage;
        const pageData = this.dataLoader.getData().pageMapping[String(pageNum)];
        if (pageData) {
            const surah = this.dataLoader.getData().suwar[pageData[1] - 1];
            document.getElementById('overlay-title').textContent = `سُورَةُ ${surah.name_ar}`;
            document.getElementById('overlay-page').textContent = `ص ${this.arabicNumerals(pageNum)}`;
        }
    }

    /**
     * Close overlay
     */
    closeOverlay() {
        document.getElementById('overlay').classList.remove('active');
    }

    /**
     * Convert to Arabic numerals
     */
    arabicNumerals(num) {
        const arabicDigits = ['٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩'];
        return String(num).split('').map(d => arabicDigits[parseInt(d)] || d).join('');
    }

    /**
     * Get juz name in Arabic words
     */
    getJuzName(juzNum) {
        const juzNames = [
            'الأَوَّل', 'الثَّانِي', 'الثَّالِث', 'الرَّابِع', 'الخَامِس',
            'السَّادِس', 'السَّابِع', 'الثَّامِن', 'التَّاسِع', 'العَاشِر',
            'الحَادِيَ عَشَرَ', 'الثَّانِيَ عَشَرَ', 'الثَّالِثَ عَشَرَ',
            'الرَّابِعَ عَشَرَ', 'الخَامِسَ عَشَرَ', 'السَّادِسَ عَشَرَ',
            'السَّابِعَ عَشَرَ', 'الثَّامِنَ عَشَرَ', 'التَّاسِعَ عَشَرَ',
            'العِشْرُونَ', 'الحَادِيَ وَالعِشْرُونَ', 'الثَّانِيَ وَالعِشْرُونَ',
            'الثَّالِثَ وَالعِشْرُونَ', 'الرَّابِعَ وَالعِشْرُونَ',
            'الخَامِسَ وَالعِشْرُونَ', 'السَّادِسَ وَالعِشْرُونَ',
            'السَّابِعَ وَالعِشْرُونَ', 'الثَّامِنَ وَالعِشْرُونَ',
            'التَّاسِعَ وَالعِشْرُونَ', 'الثَّلَاثُونَ'
        ];
        return juzNames[juzNum - 1] || this.arabicNumerals(juzNum);
    }

    // ===================================
    // TAFSEER FUNCTIONALITY
    // ===================================

    /**
     * Load tafseer data files (Arabic and English)
     */
    async loadTafseerData() {
        try {
            const [arText, enText] = await Promise.all([
                fetch('assets/tafseer/ar.muyassar.txt').then(r => r.text()),
                fetch('assets/tafseer/en.sahih.txt').then(r => r.text())
            ]);

            this.tafseerAr = this.parseTafseerFile(arText);
            this.tafseerEn = this.parseTafseerFile(enText);

            console.log('✓ Tafseer data loaded');
        } catch (error) {
            console.error('Failed to load tafseer data:', error);
        }
    }

    /**
     * Parse tafseer file (format: surah|ayah|text)
     * @returns {Map} Map with key "surah:ayah" and value as text
     */
    parseTafseerFile(text) {
        const map = new Map();
        const lines = text.trim().split('\n');

        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length >= 3) {
                const surah = parseInt(parts[0]);
                const ayah = parseInt(parts[1]);
                const content = parts.slice(2).join('|'); // In case text contains |
                map.set(`${surah}:${ayah}`, content);
            }
        }

        return map;
    }

    /**
     * Setup tafseer overlay event listeners
     */
    setupTafseerOverlay() {
        const overlay = document.getElementById('tafseerOverlay');
        const closeBtn = document.getElementById('tafseerClose');
        const playBtn = document.getElementById('tafseerPlay');

        // Play button
        if (playBtn) {
            playBtn.addEventListener('click', () => {
                const overlay = document.getElementById('tafseerOverlay');
                if (overlay.dataset.surah && overlay.dataset.ayah) {
                    this.toggleAyahAudio(overlay.dataset.surah, overlay.dataset.ayah);
                }
            });
        }

        // Close button
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeTafseer());
        }

        // Close on background click
        if (overlay) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.closeTafseer();
                }
            });
        }

        // Close on Escape key (tafseer or results overlay)
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (overlay.classList.contains('active')) {
                    this.closeTafseer();
                }
                const resultsOverlay = document.getElementById('resultsOverlay');
                if (resultsOverlay && resultsOverlay.classList.contains('active')) {
                    this.reciteController.closeResults();
                }
            }
        });
    }

    /**
     * Show tafseer for a specific ayah
     * @param {number} surah - Surah number
     * @param {number} ayah - Ayah number
     */
    showTafseer(surah, ayah) {
        const overlay = document.getElementById('tafseerOverlay');
        const titleEl = document.getElementById('tafseer-title');
        const arTextEl = document.getElementById('tafseer-ar-text');
        const enTextEl = document.getElementById('tafseer-en-text');

        if (!overlay || !this.tafseerAr || !this.tafseerEn) {
            console.warn('Tafseer not ready yet');
            return;
        }

        // Store current surah/ayah on overlay for audio playback
        overlay.dataset.surah = surah;
        overlay.dataset.ayah = ayah;

        // Reset play button
        this.updatePlayIcon(false);
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }

        const key = `${surah}:${ayah}`;
        const arContent = this.tafseerAr.get(key) || 'التفسير غير متوفر لهذه الآية';
        const enContent = this.tafseerEn.get(key) || 'Translation not available for this verse';

        // Get surah name for title
        const surahData = this.dataLoader.getSurah(surah - 1);
        const surahName = surahData ? surahData.name_ar : `سورة ${surah}`;

        // Update content
        titleEl.textContent = `${surahName} - الآية ${this.arabicNumerals(ayah)}`;
        arTextEl.textContent = arContent;
        enTextEl.textContent = enContent;

        // Show overlay
        overlay.classList.add('active');
    }

    /**
     * Close tafseer overlay
     */
    closeTafseer() {
        const overlay = document.getElementById('tafseerOverlay');
        if (overlay) {
            overlay.classList.remove('active');

            // Stop audio when closing
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
                this.updatePlayIcon(false);
            }
        }
    }

    /**
     * pad number with leading zeros (e.g. 1 -> "001")
     */
    padNumber(num) {
        return String(num).padStart(3, '0');
    }

    /**
     * Toggle audio playback for an ayah
     */
    toggleAyahAudio(surah, ayah) {
        // If already playing, stop it
        if (this.currentAudio && !this.currentAudio.paused) {
            this.currentAudio.pause();
            this.currentAudio = null;
            this.updatePlayIcon(false);
            return;
        }

        // Stop any existing audio
        if (this.currentAudio) {
            this.currentAudio.pause();
        }

        // Construct URL: https://everyayah.com/data/Alafasy_128kbps/XXXxxx.mp3
        const surahPad = this.padNumber(surah);
        const ayahPad = this.padNumber(ayah);
        const url = `https://everyayah.com/data/Alafasy_128kbps/${surahPad}${ayahPad}.mp3`;

        console.log('▶ Playing audio:', url);

        this.currentAudio = new Audio(url);

        this.currentAudio.addEventListener('play', () => {
            this.updatePlayIcon(true);
        });

        this.currentAudio.addEventListener('pause', () => {
            this.updatePlayIcon(false);
        });

        this.currentAudio.addEventListener('ended', () => {
            this.currentAudio = null;
            this.updatePlayIcon(false);
        });

        this.currentAudio.addEventListener('error', (e) => {
            console.error('Audio playback error:', e);
            this.updatePlayIcon(false);
            this.currentAudio = null;
        });

        this.currentAudio.play();
    }

    /**
     * Update play button icon
     */
    updatePlayIcon(isPlaying) {
        // Update both tafseer and results play buttons
        const btns = [
            document.getElementById('tafseerPlay'),
            document.getElementById('resultsPlay')
        ];
        for (const btn of btns) {
            if (!btn) continue;
            if (isPlaying) {
                btn.classList.add('playing');
            } else {
                btn.classList.remove('playing');
            }
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.quranApp = new QuranApp();
    window.quranApp.init();
});
