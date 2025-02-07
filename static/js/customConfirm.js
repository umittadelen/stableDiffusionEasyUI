class CustomConfirm {
    constructor() {
        this.overlay = null;
        this.box = null;
        this.isActive = false; // Tracks if a dialog is currently active
        this.escKeyListener = null; // Store reference to the keydown listener
    }

    createConfirm(message, buttons, overlayReturnValue) {
        return new Promise((resolve) => {
            // Prevent creating multiple dialogs
            if (this.isActive) {
                console.warn("A confirm dialog is already active.");
                return;
            }
            this.isActive = true;

            // Create overlay
            this.overlay = document.createElement('div');
            this.overlay.className = 'custom-confirm-overlay';

            // Create confirm box
            this.box = document.createElement('div');
            this.box.className = 'custom-confirm-box';

            // Add message
            const msg = document.createElement('p');
            message = message.replace(/\n/g, '<br>');
            msg.innerHTML = message;
            this.box.appendChild(msg);

            // Add button container
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'button-container';

            // Add buttons to the button container
            buttons.forEach((buttonConfig) => {
                const button = document.createElement('button');
                button.textContent = buttonConfig.text;
                button.addEventListener('click', () => {
                    this.closeConfirm();
                    // Execute the button's value (function) and resolve
                    if (typeof buttonConfig.value === 'function') {
                        buttonConfig.value();
                    }
                    resolve(buttonConfig.value);
                });
                buttonContainer.appendChild(button);
            });

            // Append button container to the box
            this.box.appendChild(buttonContainer);

            // Append the box to the overlay and the overlay to the document body
            this.overlay.appendChild(this.box);
            document.body.appendChild(this.overlay);

            // Force reflow to ensure the transition is applied
            window.getComputedStyle(this.overlay).opacity;

            // Add the show class to trigger the transition
            this.overlay.classList.add('show');
            this.box.classList.add('show');

            // Add overlay click listener
            this.overlay.addEventListener('click', (e) => {
                // Prevent click events from propagating when clicking the confirm box itself
                if (e.target === this.overlay) {
                    this.closeConfirm();
                    resolve(overlayReturnValue);
                }
            });

            // Add Esc key listener
            this.escKeyListener = (e) => {
                if (e.key === 'Escape') {
                    this.closeConfirm();
                    resolve(overlayReturnValue);
                }
            };
            document.addEventListener('keydown', this.escKeyListener);
        });
    }

    closeConfirm() {
        if (this.overlay) {
            this.overlay.classList.remove('show');
            this.box.classList.remove('show');
            this.overlay.addEventListener('transitionend', () => {
                if (this.overlay && this.overlay.parentNode) {
                    document.body.removeChild(this.overlay);
                    this.isActive = false; // Allow new dialogs to be created
                }
            });
        }

        // Remove Esc key listener
        if (this.escKeyListener) {
            document.removeEventListener('keydown', this.escKeyListener);
            this.escKeyListener = null;
        }
    }
}