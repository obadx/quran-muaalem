/**
 * SearchController - Manages the "Search" feature
 *
 * Flow: Record audio -> send to /search API -> navigate mushaf to matched ayah
 */
class SearchController {
    constructor(app, api) {
        this.app = app;       // QuranApp instance
        this.api = api;       // QuranAPI instance
        this.recorder = null;
        this.isSearching = false;

        // UI references (set during init)
        this.searchBtn = null;
        this.statusEl = null;
    }

    /**
     * Initialize — bind to DOM elements
     */
    init() {
        this.searchBtn = document.getElementById('searchBtn');
        this.statusEl = document.getElementById('toolbarStatus');

        if (this.searchBtn) {
            this.searchBtn.addEventListener('click', () => this.toggle());
        }
    }

    /**
     * Toggle search on/off
     */
    async toggle() {
        if (this.isSearching) {
            this.stopSearch();
        } else {
            await this.startSearch();
        }
    }

    /**
     * Start a search recording
     */
    async startSearch() {
        if (this.isSearching) return;

        try {
            this.recorder = new AudioRecorder({
                maxDuration: 15,
                onMaxDuration: () => this._onRecordingDone()
            });

            await this.recorder.start();
            this.isSearching = true;
            this._updateUI('recording');
            this._showStatus('سجّل تلاوتك...', 'recording');
        } catch (err) {
            console.error('Failed to start recording:', err);
            this._showStatus('فشل الوصول للميكروفون', 'error');
            setTimeout(() => this._clearStatus(), 3000);
        }
    }

    /**
     * Stop search recording and process
     */
    async stopSearch() {
        if (!this.isSearching) return;

        this.isSearching = false;
        this._updateUI('processing');

        if (this.recorder && this.recorder.isRecording) {
            const blob = await this.recorder.stop();
            if (blob && blob.size > 44) {
                await this._processSearchResult(blob);
            } else {
                this._showStatus('التسجيل قصير جداً', 'error');
                setTimeout(() => this._clearStatus(), 3000);
                this._updateUI('idle');
            }
        }
    }

    /**
     * Called when max duration is reached
     */
    _onRecordingDone() {
        this.isSearching = false;
        this._updateUI('processing');
        this.recorder.stop().then(blob => {
            if (blob) this._processSearchResult(blob);
        });
    }

    /**
     * Send audio to search API and navigate
     */
    async _processSearchResult(blob) {
        this._showStatus('جاري البحث...', 'processing');
        this._updateUI('processing');

        try {
            const result = await this.api.search(blob, 0.15);

            if (result.results && result.results.length > 0) {
                const match = result.results[0];
                const surah = match.start.sura_idx;
                const ayah = match.start.aya_idx;

                await this.app.goToAyah(surah, ayah, { persistent: true });

                const surahData = this.app.dataLoader.getSurah(surah - 1);
                const surahName = surahData ? surahData.name_ar : `سورة ${surah}`;
                this._showStatus(`${surahName} - الآية ${this.app.arabicNumerals(ayah)}`, 'success');
                setTimeout(() => this._clearStatus(), 4000);
            } else {
                this._showStatus(result.message || 'لم يتم العثور على تطابق', 'error');
                setTimeout(() => this._clearStatus(), 4000);
            }
        } catch (err) {
            console.error('Search error:', err);
            this._showStatus('خطأ في البحث — تأكد من تشغيل السيرفر', 'error');
            setTimeout(() => this._clearStatus(), 4000);
        }

        this._updateUI('idle');
    }

    /**
     * Update button UI state
     */
    _updateUI(state) {
        if (!this.searchBtn) return;

        this.searchBtn.classList.remove('recording', 'processing');

        switch (state) {
            case 'recording':
                this.searchBtn.classList.add('recording');
                this.searchBtn.querySelector('.btn-label').textContent = 'إيقاف';
                break;
            case 'processing':
                this.searchBtn.classList.add('processing');
                this.searchBtn.querySelector('.btn-label').textContent = 'جاري...';
                break;
            default:
                this.searchBtn.querySelector('.btn-label').textContent = 'بحث';
        }
    }

    _showStatus(text, type = '') {
        if (!this.statusEl) return;
        this.statusEl.textContent = text;
        this.statusEl.className = 'toolbar-status';
        if (type) this.statusEl.classList.add(type);
        this.statusEl.classList.add('visible');
    }

    _clearStatus() {
        if (!this.statusEl) return;
        this.statusEl.classList.remove('visible');
    }
}
