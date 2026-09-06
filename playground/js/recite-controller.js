/**
 * ReciteController - Manages the "Recite" (correction) feature
 *
 * Flow:
 * 1. User taps an ayah on the mushaf (it gets selected)
 * 2. User taps "Recite" button → recording starts
 * 3. User recites → taps stop
 * 4. Audio sent to /correct-recitation
 * 5. Errors displayed on mushaf + in results panel
 */
class ReciteController {
    constructor(app, api) {
        this.app = app;       // QuranApp instance
        this.api = api;       // QuranAPI instance
        this.recorder = null;
        this.isReciting = false;
        this.selectedAyah = null; // { surah, ayah }
        this.lastResult = null;

        // UI references
        this.reciteBtn = null;
        this.statusEl = null;
        this.resultsOverlay = null;
        this.resultsTitle = null;
        this.resultsPlayBtn = null;
    }

    /**
     * Initialize — bind to DOM elements
     */
    init() {
        this.reciteBtn = document.getElementById('reciteBtn');
        this.statusEl = document.getElementById('toolbarStatus');
        this.resultsOverlay = document.getElementById('resultsOverlay');
        this.resultsTitle = document.getElementById('results-title');
        this.resultsPlayBtn = document.getElementById('resultsPlay');

        if (this.reciteBtn) {
            this.reciteBtn.addEventListener('click', () => this.toggle());
        }

        // Setup ayah selection (click to select for recitation)
        this._setupAyahSelection();

        // Close results overlay
        const closeBtn = document.getElementById('resultsPanelClose');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeResults());
        }

        // Close on background click
        if (this.resultsOverlay) {
            this.resultsOverlay.addEventListener('click', (e) => {
                if (e.target === this.resultsOverlay) {
                    this.closeResults();
                }
            });
        }

        // Play button — plays the ayah audio
        if (this.resultsPlayBtn) {
            this.resultsPlayBtn.addEventListener('click', () => {
                if (this.resultsOverlay && this.resultsOverlay.dataset.surah && this.resultsOverlay.dataset.ayah) {
                    this.app.toggleAyahAudio(this.resultsOverlay.dataset.surah, this.resultsOverlay.dataset.ayah);
                    this._updateResultsPlayIcon();
                }
            });
        }

        // Retry / Next button — action changes dynamically based on score
        const retryBtn = document.getElementById('retryBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                if (retryBtn.dataset.action === 'next') {
                    this._nextAyah();
                } else {
                    this.closeResults();
                    this.startReciting();
                }
            });
        }
    }

    /**
     * Setup click-to-select on ayah groups
     * Reads from the app's persistentHighlight so click-to-select and
     * the mushaf highlight stay in sync.
     */
    _setupAyahSelection() {
        document.addEventListener('click', (e) => {
            const ayahGroup = e.target.closest('.ayah-group');
            if (!ayahGroup) return;

            const surah = parseInt(ayahGroup.getAttribute('data-surah'));
            const ayah = parseInt(ayahGroup.getAttribute('data-ayah'));

            if (isNaN(surah) || isNaN(ayah)) return;

            // Sync with app's persistentHighlight (set by the click handler in app.js)
            // If the same ayah was tapped again it was deselected
            if (this.app.persistentHighlight &&
                this.app.persistentHighlight.surah === surah &&
                this.app.persistentHighlight.ayah === ayah) {
                this.selectedAyah = { surah, ayah };
            } else if (!this.app.persistentHighlight) {
                this.selectedAyah = null;
            } else {
                this.selectedAyah = { surah, ayah };
            }

            this._updateReciteButton();
        });
    }

    /**
     * Toggle recite on/off
     */
    async toggle() {
        if (this.isReciting) {
            await this.stopReciting();
        } else {
            await this.startReciting();
        }
    }

    /**
     * Start recording for recitation correction
     */
    async startReciting() {
        if (this.isReciting) return;

        // Clear previous errors from mushaf
        this._clearErrorHighlights();
        this.closeResults();

        try {
            this.recorder = new AudioRecorder({
                maxDuration: 15,
                onMaxDuration: () => this._onRecordingDone()
            });

            await this.recorder.start();
            this.isReciting = true;
            this._updateUI('recording');

            // Show ayah-specific or generic message
            const target = this._getAyahTarget();
            if (target) {
                const surahData = this.app.dataLoader.getSurah(target.sura_idx - 1);
                const surahName = surahData ? surahData.name_ar : '';
                this._showStatus(`اقرأ: ${surahName} - الآية ${this.app.arabicNumerals(target.aya_idx)}`, 'recording');
            } else {
                this._showStatus('ابدأ التلاوة من أي آية', 'recording');
            }
        } catch (err) {
            console.error('Failed to start recording:', err);
            this._showStatus('فشل الوصول للميكروفون', 'error');
            setTimeout(() => this._clearStatus(), 3000);
        }
    }

    /**
     * Stop recording and submit for correction
     */
    async stopReciting() {
        if (!this.isReciting) return;

        this.isReciting = false;
        this._updateUI('processing');

        if (this.recorder && this.recorder.isRecording) {
            const blob = await this.recorder.stop();
            if (blob && blob.size > 44) {
                await this._processCorrection(blob);
            } else {
                this._showStatus('التسجيل قصير جداً', 'error');
                setTimeout(() => this._clearStatus(), 3000);
                this._updateUI('idle');
            }
        }
    }

    /**
     * Auto-process when max duration reached
     */
    _onRecordingDone() {
        this.isReciting = false;
        this._updateUI('processing');
        this.recorder.stop().then(blob => {
            if (blob) this._processCorrection(blob);
        });
    }

    /**
     * Send audio to correct-recitation API and display results
     */
    async _processCorrection(blob) {
        this._showStatus('جاري تحليل التلاوة...', 'processing');

        try {
            const ayahTarget = this._getAyahTarget();
            const result = await this.api.correctRecitation(blob, {}, ayahTarget);
            this.lastResult = result;

            // Navigate to the matched position if different
            if (result.start) {
                await this.app.goToAyah(result.start.sura_idx, result.start.aya_idx, { persistent: true });
                this.selectedAyah = { surah: result.start.sura_idx, ayah: result.start.aya_idx };
            }

            // Highlight errors on the mushaf
            this._highlightErrors(result);

            // Show results panel
            this._showResults(result);

            // Update status
            const errorCount = result.errors ? result.errors.length : 0;
            if (errorCount === 0) {
                this._showStatus('ممتاز! لا توجد أخطاء', 'success');
            } else {
                this._showStatus(`تم العثور على ${this.app.arabicNumerals(errorCount)} ${errorCount === 1 ? 'خطأ' : 'أخطاء'}`, 'error');
            }

            this._updateUI('idle');
        } catch (err) {
            console.error('Correction error:', err);
            // 500 typically means the server couldn't match the audio (incomplete recitation)
            if (err.message && err.message.includes('500')) {
                this._showStatus('لم يتم التعرف على التلاوة — حاول قراءة الآية كاملة', 'error');
            } else {
                this._showStatus('خطأ في التحليل — تأكد من تشغيل السيرفر', 'error');
            }
            setTimeout(() => this._clearStatus(), 4000);
            this._updateUI('idle');
        }
    }

    /**
     * Highlight error words on the mushaf
     * Maps uthmani_pos from API response back to DOM glyph elements
     */
    _highlightErrors(result) {
        this._clearErrorHighlights();

        if (!result.errors || result.errors.length === 0) return;
        if (!result.start) return;

        const surah = result.start.sura_idx;
        const ayah = result.start.aya_idx;

        // Get all word glyphs for this ayah on the current page
        const ayahGlyphs = document.querySelectorAll(
            `.ayah-group[data-surah="${surah}"][data-ayah="${ayah}"] .qcf.word`
        );

        if (ayahGlyphs.length === 0) return;

        // Build a word-to-character-range map from the uthmani text
        // Each word in the text maps to a glyph in the DOM
        const words = result.uthmani_text ? result.uthmani_text.split(/\s+/) : [];
        const wordRanges = []; // { start, end } character positions for each word
        let charPos = 0;
        for (const word of words) {
            wordRanges.push({ start: charPos, end: charPos + word.length });
            charPos += word.length + 1; // +1 for the space
        }

        for (const error of result.errors) {
            const [startPos, endPos] = error.uthmani_pos || [0, 0];

            // Find which words overlap with this error's character range
            ayahGlyphs.forEach((glyph, idx) => {
                if (idx >= wordRanges.length) return;

                const wordStart = wordRanges[idx].start;
                const wordEnd = wordRanges[idx].end;

                // Check if this word overlaps with the error range
                if (wordStart < endPos && wordEnd > startPos) {
                    const errorClass = error.error_type === 'tajweed' ? 'error-tajweed' : 'error-speech';
                    glyph.classList.add('error-highlight', errorClass);
                }
            });
        }
    }

    /**
     * Clear all error highlights from mushaf
     */
    _clearErrorHighlights() {
        document.querySelectorAll('.error-highlight').forEach(el => {
            el.classList.remove('error-highlight', 'error-tajweed', 'error-speech');
        });
    }

    /**
     * Build and show the results panel
     */
    _showResults(result) {
        if (!this.resultsOverlay) return;

        // Set title with surah name + ayah number
        if (result.start && this.resultsTitle) {
            const surahData = this.app.dataLoader.getSurah(result.start.sura_idx - 1);
            const surahName = surahData ? surahData.name_ar : `سورة ${result.start.sura_idx}`;
            this.resultsTitle.textContent = `${surahName} - الآية ${this.app.arabicNumerals(result.start.aya_idx)}`;

            // Store surah/ayah on overlay for play button
            this.resultsOverlay.dataset.surah = result.start.sura_idx;
            this.resultsOverlay.dataset.ayah = result.start.aya_idx;
        }

        // Reset play button
        if (this.resultsPlayBtn) {
            this.resultsPlayBtn.classList.remove('playing');
        }

        const errors = result.errors || [];
        const totalPhonemes = result.reference_phonemes ? result.reference_phonemes.length : 1;
        const errorPhonemes = errors.reduce((sum, e) => sum + (e.expected_len || 1), 0);
        const score = Math.max(0, Math.round((1 - errorPhonemes / totalPhonemes) * 100));

        // Update retry/next button based on score
        this._updateRetryButton(errors.length === 0);

        // Score color
        let scoreClass = 'score-excellent';
        if (score < 60) scoreClass = 'score-poor';
        else if (score < 80) scoreClass = 'score-fair';
        else if (score < 95) scoreClass = 'score-good';

        // Build word ranges from uthmani text
        const words = result.uthmani_text ? result.uthmani_text.split(/\s+/) : [];
        const wordRanges = [];
        let cp = 0;
        for (const w of words) {
            wordRanges.push({ start: cp, end: cp + w.length, text: w });
            cp += w.length + 1;
        }

        // Build errors HTML — grouped by word
        let errorsHTML = '';
        if (errors.length === 0) {
            errorsHTML = `
                <div class="result-perfect">
                    <span class="perfect-icon">&#x2714;</span>
                    <p>ما شاء الله — تلاوة صحيحة!</p>
                </div>
            `;
        } else {
            // Group errors by word index
            const wordErrors = new Map(); // wordIdx -> [error, ...]
            for (const error of errors) {
                const [startPos, endPos] = error.uthmani_pos || [0, 0];
                let assigned = false;
                wordRanges.forEach((range, idx) => {
                    if (range.start < endPos && range.end > startPos) {
                        if (!wordErrors.has(idx)) wordErrors.set(idx, []);
                        wordErrors.get(idx).push(error);
                        assigned = true;
                    }
                });
                // If no word matched, group under -1
                if (!assigned) {
                    if (!wordErrors.has(-1)) wordErrors.set(-1, []);
                    wordErrors.get(-1).push(error);
                }
            }

            // Build HTML for each word group
            const groupsHTML = [];
            for (const [wordIdx, wordErrs] of wordErrors) {
                const wordText = wordIdx >= 0 && wordIdx < words.length ? words[wordIdx] : '';

                const errItems = wordErrs.map(error => {
                    const ruleName = error.ref_tajweed_rules && error.ref_tajweed_rules.length > 0
                        ? error.ref_tajweed_rules[0].name
                        : null;

                    const ruleAr = ruleName ? ruleName.ar : (error.error_type === 'tajweed' ? 'خطأ تجويد' : 'خطأ نطق');
                    const ruleEn = ruleName ? ruleName.en : (error.error_type === 'tajweed' ? 'Tajweed Error' : 'Speech Error');

                    const errorTypeClass = error.error_type === 'tajweed' ? 'tag-tajweed' : 'tag-speech';

                    // Phoneme comparison (expected vs predicted)
                    let phonemeHTML = '';
                    if (error.expected_ph || error.preditected_ph) {
                        const expected = error.expected_ph || '—';
                        const predicted = error.preditected_ph || '—';
                        phonemeHTML = `
                            <div class="phoneme-compare">
                                <span class="phoneme-expected">${expected}</span>
                                <span class="phoneme-arrow">→</span>
                                <span class="phoneme-predicted">${predicted}</span>
                            </div>
                        `;
                    }

                    let detailText = '';
                    if (error.expected_len && error.predicted_len) {
                        detailText = `المتوقع: ${error.expected_len} حروف — المنطوق: ${error.predicted_len} حروف`;
                    }

                    return `
                        <div class="error-item-sub">
                            <div class="error-item-header">
                                <span class="error-tag ${errorTypeClass}">${ruleAr}</span>
                                <span class="error-tag-en">${ruleEn}</span>
                            </div>
                            ${phonemeHTML}
                            ${detailText ? `<p class="error-detail">${detailText}</p>` : ''}
                        </div>
                    `;
                }).join('');

                const borderClass = wordErrs[0].error_type === 'tajweed' ? 'word-group-tajweed' : 'word-group-speech';

                groupsHTML.push(`
                    <div class="error-word-group ${borderClass}">
                        ${wordText ? `<div class="error-word-title">${wordText}</div>` : ''}
                        ${errItems}
                    </div>
                `);
            }
            errorsHTML = groupsHTML.join('');
        }

        // Build uthmani text with error words highlighted
        let uthmaniHTML = result.uthmani_text || '';
        if (words.length > 0 && errors.length > 0) {
            const errorWordIndices = new Set();
            for (const error of errors) {
                const [startPos, endPos] = error.uthmani_pos || [0, 0];
                wordRanges.forEach((range, idx) => {
                    if (range.start < endPos && range.end > startPos) {
                        errorWordIndices.add(idx);
                    }
                });
            }

            uthmaniHTML = words.map((word, idx) => {
                if (errorWordIndices.has(idx)) {
                    return `<span class="uthmani-error-word">${word}</span>`;
                }
                return word;
            }).join(' ');
        }

        // Build panel content
        const panelContent = document.getElementById('resultsPanelContent');
        if (panelContent) {
            panelContent.innerHTML = `
                <div class="results-score ${scoreClass}">
                    <span class="score-number">${score}%</span>
                    <span class="score-label">${score >= 95 ? 'ممتاز' : score >= 80 ? 'جيد' : score >= 60 ? 'مقبول' : 'يحتاج تحسين'}</span>
                </div>
                <div class="results-text">
                    <p class="uthmani-text">${uthmaniHTML}</p>
                </div>
                <div class="errors-list">
                    ${errorsHTML}
                </div>
            `;
        }

        this.resultsOverlay.classList.add('active');
    }

    /**
     * Close the results panel
     */
    closeResults() {
        if (this.resultsOverlay) {
            this.resultsOverlay.classList.remove('active');
        }
        // Stop audio if playing
        if (this.app.currentAudio) {
            this.app.currentAudio.pause();
            this.app.currentAudio = null;
        }
        if (this.resultsPlayBtn) {
            this.resultsPlayBtn.classList.remove('playing');
        }
        this._clearErrorHighlights();
    }

    /**
     * Update the recite button state based on selection
     */
    _updateReciteButton() {
        if (!this.reciteBtn) return;

        if (this.selectedAyah) {
            this.reciteBtn.classList.add('has-selection');
        } else {
            this.reciteBtn.classList.remove('has-selection');
        }
    }

    /**
     * Update button UI state
     */
    _updateUI(state) {
        if (!this.reciteBtn) return;

        this.reciteBtn.classList.remove('recording', 'processing');

        switch (state) {
            case 'recording':
                this.reciteBtn.classList.add('recording');
                this.reciteBtn.querySelector('.btn-label').textContent = 'إيقاف';
                break;
            case 'processing':
                this.reciteBtn.classList.add('processing');
                this.reciteBtn.querySelector('.btn-label').textContent = 'جاري...';
                break;
            default:
                this.reciteBtn.querySelector('.btn-label').textContent = 'تلاوة';
        }
    }

    /**
     * Get the ayah target by finding the highlighted ayah in the DOM.
     * Returns { sura_idx, aya_idx } or null.
     */
    _getAyahTarget() {
        // Find the highlighted ayah directly from the DOM
        const el = document.querySelector('.ayah-group.persistent-highlight');
        if (!el) return null;

        const surah = parseInt(el.getAttribute('data-surah'));
        const ayah = parseInt(el.getAttribute('data-ayah'));
        if (isNaN(surah) || isNaN(ayah)) return null;

        return { sura_idx: surah, aya_idx: ayah };
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

    /**
     * Swap the retry button to "next" arrow or back to "retry"
     */
    _updateRetryButton(isPerfect) {
        const btn = document.getElementById('retryBtn');
        if (!btn) return;

        if (isPerfect) {
            btn.dataset.action = 'next';
            btn.setAttribute('aria-label', 'الآية التالية');
            // Next arrow SVG (forward arrow)
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                <line x1="5" y1="12" x2="19" y2="12"/>
                <polyline points="12 5 19 12 12 19"/>
            </svg>`;
        } else {
            btn.dataset.action = 'retry';
            btn.setAttribute('aria-label', 'إعادة');
            // Retry arrow SVG
            btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
                <polyline points="23 4 23 10 17 10"/>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
            </svg>`;
        }
    }

    /**
     * Navigate to the next ayah and start recording
     */
    async _nextAyah() {
        if (!this.lastResult || !this.lastResult.start) return;

        let surah = this.lastResult.start.sura_idx;
        let ayah = this.lastResult.start.aya_idx;

        // Get current surah's ayah count
        const surahData = this.app.dataLoader.getSurah(surah - 1);
        const ayahCount = surahData ? surahData.ayah_count : 0;

        // Advance to next ayah
        ayah++;
        if (ayah > ayahCount) {
            // Move to next surah
            surah++;
            ayah = 1;
            if (surah > 114) {
                // Wrap to beginning of Quran
                surah = 1;
                ayah = 1;
            }
        }

        // Close results and navigate
        this.closeResults();
        await this.app.goToAyah(surah, ayah, { persistent: true });
        this.selectedAyah = { surah, ayah };

        // Start recording for the next ayah
        this.startReciting();
    }

    /**
     * Sync the results play button icon with audio state
     */
    _updateResultsPlayIcon() {
        if (!this.resultsPlayBtn) return;
        // Check after a short delay to let audio state update
        setTimeout(() => {
            if (this.app.currentAudio && !this.app.currentAudio.paused) {
                this.resultsPlayBtn.classList.add('playing');
            } else {
                this.resultsPlayBtn.classList.remove('playing');
            }
        }, 100);
    }
}
