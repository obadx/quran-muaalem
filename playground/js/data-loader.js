/**
 * Data Loader Module for New App
 * Loads Quran data from external JSON/CSV files (with correct relative paths)
 */

class QuranDataLoader {
    constructor(dataPath = 'data') {
        this.dataPath = dataPath;
        this.suwar = null;
        this.pageMapping = null;
        this.mushafMetadata = null;
        this.layoutData = null;
        this.loading = false;
        this.loaded = false;
    }

    /**
     * Load all data files
     * @returns {Promise<Object>} All loaded data
     */
    async loadAll() {
        if (this.loaded) {
            return this.getData();
        }

        if (this.loading) {
            // Wait for existing load to complete
            return new Promise((resolve) => {
                const checkLoaded = setInterval(() => {
                    if (this.loaded) {
                        clearInterval(checkLoaded);
                        resolve(this.getData());
                    }
                }, 100);
            });
        }

        this.loading = true;

        try {
            // Load all JSON files in parallel
            const [suwar, pageMapping, mushafMetadata, layoutCSV] = await Promise.all([
                fetch(`${this.dataPath}/suwar.json`).then((r) => r.json()),
                fetch(`${this.dataPath}/page_mapping.json`).then((r) => r.json()),
                fetch(`${this.dataPath}/mushaf_metadata.json`).then((r) => r.json()),
                fetch(`${this.dataPath}/quran_layout.csv`).then((r) => r.text()),
            ]);

            this.suwar = suwar.suwar || suwar; // Extract array from wrapper
            this.pageMapping = pageMapping;
            this.mushafMetadata = mushafMetadata;
            this.layoutData = this.parseCSV(layoutCSV);

            this.loaded = true;
            this.loading = false;

            return this.getData();
        } catch (error) {
            this.loading = false;
            console.error('Failed to load Quran data:', error);
            throw error;
        }
    }

    /**
     * Parse CSV layout data into structured format
     * @param {string} csv - CSV string
     * @returns {Array} Parsed data rows
     */
    parseCSV(csv) {
        const lines = csv.trim().split('\n');
        const headers = lines[0].split(',');
        const data = [];

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            if (values.length === headers.length) {
                const row = {
                    sura: parseInt(values[0]),
                    verse: parseInt(values[1]),
                    pageNo: parseInt(values[2]),
                    lineNo: parseInt(values[3]),
                    fontFile: parseInt(values[4]),
                    fontCode: parseInt(values[5]),
                    type: parseInt(values[6]),
                };
                data.push(row);
            }
        }

        return data;
    }

    /**
     * Get all loaded data
     * @returns {Object} All data
     */
    getData() {
        return {
            suwar: this.suwar,
            pageMapping: this.pageMapping,
            mushafMetadata: this.mushafMetadata,
            layoutData: this.layoutData,
            // For backward compatibility with old variable names
            as: this.suwar,
            os: this.pageMapping,
            ss: this.mushafMetadata,
            fs: this.layoutData,
        };
    }

    /**
     * Get layout data for a specific page
     * @param {number} pageNo - Page number (1-604)
     * @returns {Array} Layout data for the page
     */
    getPageLayout(pageNo) {
        if (!this.layoutData) {
            throw new Error('Data not loaded yet. Call loadAll() first.');
        }
        return this.layoutData.filter((row) => row.pageNo === pageNo);
    }

    /**
     * Get surah info by index (0-based)
     * @param {number} index - Surah index (0-113)
     * @returns {Object} Surah info
     */
    getSurah(index) {
        if (!this.suwar) {
            throw new Error('Data not loaded yet. Call loadAll() first.');
        }
        return this.suwar[index];
    }

    /**
     * Get page info by page number
     * @param {number} pageNo - Page number (1-604)
     * @returns {Array} [surah, startAyah, startPage, endAyah]
     */
    getPageInfo(pageNo) {
        if (!this.pageMapping) {
            throw new Error('Data not loaded yet. Call loadAll() first.');
        }
        return this.pageMapping[pageNo.toString()];
    }
}

// For browser
if (typeof window !== 'undefined') {
    window.QuranDataLoader = QuranDataLoader;
}
