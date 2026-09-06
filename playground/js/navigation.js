/**
 * NavigationBar - Handles all navigation controls
 */
class NavigationBar {
    constructor(app) {
        this.app = app;
        this.currentPage = 1;
        this.totalPages = 604;
    }

    /**
     * Initialize navigation with DOM elements
     */
    init() {
        // Get elements
        this.prevBtn = document.getElementById('prevPage');
        this.nextBtn = document.getElementById('nextPage');
        this.pageInfo = document.getElementById('pageInfo');
        this.pageInput = document.getElementById('pageInput');
        this.surahSelect = document.getElementById('surahSelect');
        this.juzSelect = document.getElementById('juzSelect');

        // Setup event listeners
        this.setupEventListeners();

        // Populate selectors
        this.populateSelectors();
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Page navigation buttons
        this.prevBtn.addEventListener('click', () => this.goToPreviousPage());
        this.nextBtn.addEventListener('click', () => this.goToNextPage());

        // Page input
        this.pageInput.addEventListener('change', (e) => {
            const page = parseInt(e.target.value);
            if (page >= 1 && page <= this.totalPages) {
                this.goToPage(page);
            }
        });

        // Surah selector
        this.surahSelect.addEventListener('change', (e) => {
            const index = parseInt(e.target.value);
            if (!isNaN(index)) {
                const surah = this.app.dataLoader.getSurah(index);
                this.goToPage(surah.start_page);
            }
        });

        // Juz selector
        this.juzSelect.addEventListener('change', (e) => {
            const juzIndex = parseInt(e.target.value);
            if (!isNaN(juzIndex)) {
                const metadata = this.app.dataLoader.getData().mushafMetadata;
                const juzPage = metadata.juz_pages[juzIndex];
                this.goToPage(juzPage);
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                e.preventDefault();
                this.goToPreviousPage();
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                e.preventDefault();
                this.goToNextPage();
            }
        });
    }

    /**
     * Populate surah and juz selectors
     */
    populateSelectors() {
        const data = this.app.dataLoader.getData();

        // Populate surah selector
        if (data.suwar) {
            data.suwar.forEach((surah, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${index + 1}. ${surah.name_ar}`;
                this.surahSelect.appendChild(option);
            });
        }

        // Populate juz selector
        if (data.mushafMetadata && data.mushafMetadata.juz_pages) {
            for (let i = 0; i < 30; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `الجزء ${i + 1}`;
                this.juzSelect.appendChild(option);
            }
        }
    }

    /**
     * Navigate to specific page
     */
    goToPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.totalPages) {
            return;
        }

        this.currentPage = pageNumber;
        this.app.renderPage(pageNumber);
        this.updateUI();
    }

    /**
     * Go to previous page
     */
    goToPreviousPage() {
        if (this.currentPage > 1) {
            this.goToPage(this.currentPage - 1);
        }
    }

    /**
     * Go to next page
     */
    goToNextPage() {
        if (this.currentPage < this.totalPages) {
            this.goToPage(this.currentPage + 1);
        }
    }

    /**
     * Update UI elements
     */
    updateUI() {
        // Update page info
        this.pageInfo.textContent = `صفحة ${this.currentPage} من ${this.totalPages}`;

        // Update page input
        this.pageInput.value = this.currentPage;

        // Update button states
        this.prevBtn.disabled = this.currentPage === 1;
        this.nextBtn.disabled = this.currentPage === this.totalPages;

        // Update URL hash
        window.location.hash = `page/${this.currentPage}`;
    }

    /**
     * Get current page number
     */
    getCurrentPage() {
        return this.currentPage;
    }
}

// Export for ES6 modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NavigationBar;
}
