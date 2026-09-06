/**
 * AudioRecorder - Records microphone audio as 16kHz mono WAV
 *
 * Uses Web Audio API + ScriptProcessorNode to capture raw PCM,
 * then encodes to WAV blob. No external dependencies.
 *
 * Usage:
 *   const recorder = new AudioRecorder();
 *   await recorder.start();
 *   const blob = await recorder.stop(); // WAV Blob
 */
class AudioRecorder {
    constructor(options = {}) {
        this.targetSampleRate = options.sampleRate || 16000;
        this.maxDuration = options.maxDuration || 15; // seconds
        this.onMaxDuration = options.onMaxDuration || null;

        // Internal state
        this._stream = null;
        this._audioContext = null;
        this._sourceNode = null;
        this._processorNode = null;
        this._chunks = [];
        this._isRecording = false;
        this._startTime = null;
        this._stopResolve = null;
    }

    get isRecording() {
        return this._isRecording;
    }

    get elapsed() {
        if (!this._startTime) return 0;
        return (Date.now() - this._startTime) / 1000;
    }

    /**
     * Request microphone permission and start recording
     */
    async start() {
        if (this._isRecording) {
            throw new Error('Already recording');
        }

        // Request microphone
        this._stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: { ideal: this.targetSampleRate },
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: true
            }
        });

        // Create audio context at native sample rate (we'll resample later)
        this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const nativeSampleRate = this._audioContext.sampleRate;

        this._sourceNode = this._audioContext.createMediaStreamSource(this._stream);

        // Use ScriptProcessorNode to capture raw PCM
        // Buffer size of 4096 is a good balance between latency and performance
        this._processorNode = this._audioContext.createScriptProcessor(4096, 1, 1);
        this._chunks = [];
        this._isRecording = true;
        this._startTime = Date.now();

        this._processorNode.onaudioprocess = (e) => {
            if (!this._isRecording) return;

            const inputData = e.inputBuffer.getChannelData(0);
            // Store a copy of the float32 samples
            this._chunks.push(new Float32Array(inputData));

            // Auto-stop at max duration
            if (this.elapsed >= this.maxDuration) {
                if (this.onMaxDuration) {
                    this.onMaxDuration();
                }
                this.stop();
            }
        };

        this._sourceNode.connect(this._processorNode);
        this._processorNode.connect(this._audioContext.destination);

        console.log(`Recording started (native: ${nativeSampleRate}Hz, target: ${this.targetSampleRate}Hz)`);
    }

    /**
     * Stop recording and return WAV blob
     * @returns {Promise<Blob>} WAV audio blob at 16kHz mono
     */
    stop() {
        return new Promise((resolve) => {
            if (!this._isRecording) {
                resolve(null);
                return;
            }

            this._isRecording = false;

            // Disconnect audio nodes
            if (this._processorNode) {
                this._processorNode.disconnect();
                this._processorNode = null;
            }
            if (this._sourceNode) {
                this._sourceNode.disconnect();
                this._sourceNode = null;
            }

            // Stop microphone
            if (this._stream) {
                this._stream.getTracks().forEach(t => t.stop());
                this._stream = null;
            }

            const nativeSampleRate = this._audioContext.sampleRate;

            // Close audio context
            if (this._audioContext) {
                this._audioContext.close();
                this._audioContext = null;
            }

            // Merge all chunks into a single Float32Array
            const totalLength = this._chunks.reduce((acc, c) => acc + c.length, 0);
            const merged = new Float32Array(totalLength);
            let offset = 0;
            for (const chunk of this._chunks) {
                merged.set(chunk, offset);
                offset += chunk.length;
            }
            this._chunks = [];

            // Resample to target sample rate
            const resampled = this._resample(merged, nativeSampleRate, this.targetSampleRate);

            // Encode as WAV
            const wavBlob = this._encodeWAV(resampled, this.targetSampleRate);

            const duration = (resampled.length / this.targetSampleRate).toFixed(1);
            console.log(`Recording stopped: ${duration}s, ${wavBlob.size} bytes`);

            resolve(wavBlob);
        });
    }

    /**
     * Cancel recording without returning data
     */
    cancel() {
        this._isRecording = false;
        this._chunks = [];

        if (this._processorNode) {
            this._processorNode.disconnect();
            this._processorNode = null;
        }
        if (this._sourceNode) {
            this._sourceNode.disconnect();
            this._sourceNode = null;
        }
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }
        if (this._audioContext) {
            this._audioContext.close();
            this._audioContext = null;
        }

        console.log('Recording cancelled');
    }

    /**
     * Simple linear interpolation resampler
     */
    _resample(samples, fromRate, toRate) {
        if (fromRate === toRate) return samples;

        const ratio = fromRate / toRate;
        const newLength = Math.round(samples.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcFloor = Math.floor(srcIndex);
            const srcCeil = Math.min(srcFloor + 1, samples.length - 1);
            const frac = srcIndex - srcFloor;
            result[i] = samples[srcFloor] * (1 - frac) + samples[srcCeil] * frac;
        }

        return result;
    }

    /**
     * Encode Float32Array as 16-bit PCM WAV blob
     */
    _encodeWAV(samples, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;
        const dataSize = samples.length * blockAlign;
        const headerSize = 44;
        const buffer = new ArrayBuffer(headerSize + dataSize);
        const view = new DataView(buffer);

        // RIFF header
        this._writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        this._writeString(view, 8, 'WAVE');

        // fmt chunk
        this._writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true); // chunk size
        view.setUint16(20, 1, true);  // PCM format
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true); // byte rate
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);

        // data chunk
        this._writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        // Write PCM samples (float32 -> int16)
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
            view.setInt16(offset, val, true);
            offset += 2;
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    _writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }
}
