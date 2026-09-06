/**
 * QuranAPI - Client for the quran-muaalem API server
 *
 * Default '/api' goes through the nginx reverse proxy.
 * For local dev with server.py, also use '/api'.
 */
class QuranAPI {
    constructor(baseURL = null) {
        if (baseURL !== null) {
            this.baseURL = baseURL;
        } else {
            // Auto-detect: when served directly from the API server (port 8001),
            // use root paths. Otherwise (e.g., behind nginx proxy), use /api prefix.
            this.baseURL = window.location.port === '8001' ? '' : '/api';
        }
    }

    /**
     * Check if the API server is healthy and engine is connected
     * @returns {Promise<{status: string, engine_status: string}>}
     */
    async health() {
        const resp = await fetch(`${this.baseURL}/health`);
        if (!resp.ok) throw new Error(`Health check failed: ${resp.status}`);
        return resp.json();
    }

    /**
     * Search the Quran by audio recording
     * @param {Blob} audioBlob - WAV audio blob
     * @param {number} errorRatio - Fuzzy match tolerance (0.0-1.0), default 0.15
     * @returns {Promise<SearchResult>}
     *
     * Response shape:
     * {
     *   phonemes: string,
     *   results: [{
     *     start: { sura_idx, aya_idx, uthmani_word_idx, uthmani_char_idx, phonemes_idx },
     *     end:   { sura_idx, aya_idx, uthmani_word_idx, uthmani_char_idx, phonemes_idx },
     *     uthmani_text: string
     *   }],
     *   message: string|null
     * }
     */
    async search(audioBlob, errorRatio = 0.15) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');

        const url = `${this.baseURL}/search?error_ratio=${errorRatio}`;
        const resp = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`Search failed (${resp.status}): ${err}`);
        }

        return resp.json();
    }

    /**
     * Correct recitation — analyze audio for tajweed errors
     * @param {Blob} audioBlob - WAV audio blob
     * @param {Object} options - Optional tajweed parameters
     * @returns {Promise<CorrectionResult>}
     *
     * Response shape:
     * {
     *   start: { sura_idx, aya_idx, ... },
     *   end: { sura_idx, aya_idx, ... },
     *   predicted_phonemes: string,
     *   reference_phonemes: string,
     *   uthmani_text: string,
     *   errors: [{
     *     uthmani_pos: [start, end],
     *     ph_pos: [start, end],
     *     error_type: "tajweed"|"speech",
     *     speech_error_type: "replace"|"insert"|"delete",
     *     expected_ph: string,
     *     preditected_ph: string,
     *     expected_len: number,
     *     predicted_len: number,
     *     ref_tajweed_rules: [{
     *       name: { ar: string, en: string },
     *       golden_len: number,
     *       correctness_type: string,
     *       tag: string
     *     }]
     *   }]
     * }
     */
    async correctRecitation(audioBlob, options = {}, ayahTarget = null) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');

        // If ayah is specified, pass it to skip the search
        if (ayahTarget && ayahTarget.sura_idx && ayahTarget.aya_idx) {
            formData.append('sura_idx', String(ayahTarget.sura_idx));
            formData.append('aya_idx', String(ayahTarget.aya_idx));
        }

        // Append optional tajweed parameters
        const defaults = {
            rewaya: 'hafs',
            recitation_speed: 'murattal',
            madd_monfasel_len: 4,
            madd_mottasel_len: 4,
            madd_mottasel_waqf: 4,
            madd_aared_len: 4
        };

        const params = { ...defaults, ...options };
        for (const [key, value] of Object.entries(params)) {
            if (value !== null && value !== undefined) {
                formData.append(key, String(value));
            }
        }

        const resp = await fetch(`${this.baseURL}/correct-recitation`, {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`Correction failed (${resp.status}): ${err}`);
        }

        return resp.json();
    }

    /**
     * Transcribe audio to phonemes (proxy to engine)
     * @param {Blob} audioBlob - WAV audio blob
     * @returns {Promise<{phonemes: string, sifat: any}>}
     */
    async transcript(audioBlob) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');

        const resp = await fetch(`${this.baseURL}/transcript`, {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`Transcript failed (${resp.status}): ${err}`);
        }

        return resp.json();
    }
}
