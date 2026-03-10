// --- Binary unpacking ---
function unpackBinaryData(packedData, width, height, frameCount, framePackedSize) {
    const frames = new Array(frameCount);
    for (let f = 0; f < frameCount; f++) {
        const frame = new Array(height);
        const frameOffset = f * framePackedSize;
        for (let y = 0; y < height; y++) {
            const row = new Array(width);
            frame[y] = row;
            for (let x = 0; x < width; x++) {
                const bitPos = y * width + x;
                const bytePos = frameOffset + (bitPos >> 3);
                const bitOffset = 7 - (bitPos & 7);
                row[x] = ((packedData[bytePos] >> bitOffset) & 1) * 255;
            }
        }
        frames[f] = frame;
    }
    return frames;
}

// --- CaptchaInstance: one independent captcha card ---
class CaptchaInstance {
    constructor(card) {
        this.card = card;
        this.canvas = card.querySelector('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.mode = this.canvas.dataset.mode;
        this.captchaId = null;
        this.frames = [];
        this.currentFrameIndex = 0;
        this.lastFrameTime = 0;
        this.animationFrameId = null;
        this.imageData = null;
        this.imageDataBuffer = null;
        this.frameInterval = 1000 / 60;
        this.startFrame = 0;
        this.isPaused = false;

        // Bind UI
        card.querySelector('[data-verify]').addEventListener('click', () => this.verify());
        card.querySelector('[data-refresh]').addEventListener('click', () => this.generate());
        card.querySelector('[data-input]').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.verify();
        });
        card.querySelector('[data-pause]').addEventListener('click', () => {
            this.isPaused = !this.isPaused;
            card.querySelector('[data-pause]').textContent = this.isPaused ? '\u25B6' : '\u275A\u275A';
        });
    }

    animate(currentTime) {
        if (this.isPaused) {
            this.animationFrameId = requestAnimationFrame((t) => this.animate(t));
            return;
        }
        if (!this.lastFrameTime) {
            this.lastFrameTime = currentTime;
            this.animationFrameId = requestAnimationFrame((t) => this.animate(t));
            return;
        }
        const elapsed = currentTime - this.lastFrameTime;
        if (elapsed >= this.frameInterval && this.frames.length > 0) {
            const frame = this.frames[this.currentFrameIndex];
            if (frame) {
                const pixelSize = 2;
                this.imageDataBuffer.fill(0);
                for (let y = 0; y < frame.length; y++) {
                    const row = frame[y];
                    for (let x = 0; x < row.length; x++) {
                        const val = row[x];
                        let pixel;
                        if (val === 0) {
                            pixel = (255 << 24) | (82 << 16) | (112 << 8) | 90;
                        } else {
                            pixel = (255 << 24) | (val << 16) | (val << 8) | val;
                        }
                        for (let py = 0; py < pixelSize; py++) {
                            const bufferY = (y * pixelSize + py) * this.canvas.width;
                            for (let px = 0; px < pixelSize; px++) {
                                this.imageDataBuffer[bufferY + (x * pixelSize + px)] = pixel;
                            }
                        }
                    }
                }
                this.ctx.putImageData(this.imageData, 0, 0);
            }
            this.currentFrameIndex = (this.currentFrameIndex + 1) % this.frames.length;
            this.lastFrameTime = currentTime - (elapsed % this.frameInterval);
        }
        this.animationFrameId = requestAnimationFrame((t) => this.animate(t));
    }

    startAnimation() {
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
        this.currentFrameIndex = this.startFrame % Math.max(this.frames.length, 1);
        this.lastFrameTime = 0;
        this.canvas.style.opacity = '1';
        this.animate(performance.now());
    }

    async generate() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        this.canvas.style.opacity = '0';
        this.frames = [];
        this.currentFrameIndex = 0;
        this.lastFrameTime = 0;
        this.isPaused = false;
        this.card.querySelector('[data-pause]').textContent = '\u275A\u275A';

        const result = this.card.querySelector('[data-result]');
        result.textContent = '';
        result.className = 'result';
        this.card.querySelector('[data-hints-display]').innerHTML = '';

        try {
            const response = await fetch('/api/captcha/generate-all', {
                method: 'POST',
                headers: { 'Accept': 'application/octet-stream', 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: this.mode })
            });

            const width = parseInt(response.headers.get('X-Frame-Width') || response.headers.get('x-frame-width'), 10);
            const height = parseInt(response.headers.get('X-Frame-Height') || response.headers.get('x-frame-height'), 10);
            const totalFrameCount = parseInt(response.headers.get('X-Total-Frame-Count') || response.headers.get('x-total-frame-count'), 10);
            const framePackedSize = parseInt(response.headers.get('X-Frame-Packed-Size') || response.headers.get('x-frame-packed-size'), 10);
            this.captchaId = response.headers.get('X-Captcha-Id') || response.headers.get('x-captcha-id');
            this.frameInterval = parseFloat(response.headers.get('X-Frame-Interval') || response.headers.get('x-frame-interval')) || (1000 / 60);
            this.startFrame = parseInt(response.headers.get('X-Start-Frame') || response.headers.get('x-start-frame') || '0', 10);

            this.canvas.width = width * 2;
            this.canvas.height = height * 2;
            this.imageData = this.ctx.createImageData(this.canvas.width, this.canvas.height);
            this.imageDataBuffer = new Uint32Array(this.imageData.data.buffer);

            const allBuffer = await response.arrayBuffer();
            this.frames = unpackBinaryData(new Uint8Array(allBuffer), width, height, totalFrameCount, framePackedSize);

            this.startAnimation();
            const input = this.card.querySelector('[data-input]');
            input.value = '';
        } catch (error) {
            console.error(`Error generating ${this.mode} CAPTCHA:`, error);
            result.textContent = 'Error loading CAPTCHA.';
            result.className = 'result error';
        }
    }

    renderHints(hints) {
        const container = this.card.querySelector('[data-hints-display]');
        container.innerHTML = '';
        if (!hints || !hints.length) return;
        hints.forEach(h => {
            const el = document.createElement('div');
            el.className = 'hint-letter hint-' + h.status;
            el.textContent = h.letter;
            container.appendChild(el);
        });
    }

    async verify() {
        const input = this.card.querySelector('[data-input]');
        const result = this.card.querySelector('[data-result]');
        const withHints = this.card.querySelector('[data-hints]').checked;
        const text = input.value.trim().toUpperCase();

        if (!this.captchaId) {
            result.textContent = 'No active CAPTCHA. Click "New".';
            result.className = 'result error';
            return;
        }
        if (!text || text.length !== 6) {
            result.textContent = 'Enter 6 characters';
            result.className = 'result error';
            input.focus();
            return;
        }

        try {
            const response = await fetch('/api/captcha/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    captcha_id: this.captchaId,
                    text: text,
                    honeypot: '',
                    with_hints: withHints
                })
            });
            const data = await response.json();

            if (data.success) {
                result.textContent = 'Correct!';
                result.className = 'result success';
                this.card.querySelector('[data-hints-display]').innerHTML = '';
            } else {
                result.textContent = data.error || 'Incorrect.';
                result.className = 'result error';
                if (data.attempts_remaining != null) {
                    result.textContent += ` (${data.attempts_remaining} left)`;
                }
                if (data.hints) {
                    this.renderHints(data.hints);
                } else {
                    this.card.querySelector('[data-hints-display]').innerHTML = '';
                }
                input.value = '';
                input.focus();
            }
        } catch (error) {
            console.error('Verify error:', error);
            result.textContent = 'Error verifying.';
            result.className = 'result error';
        }
    }
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.captcha-card');
    cards.forEach(card => {
        const instance = new CaptchaInstance(card);
        instance.generate();
    });
});
