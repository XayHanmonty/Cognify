const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
let currentMode = 'chat'; // Default mode

// Function to auto-resize textarea
function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
}

// Add input event listener for auto-resize
userInput.addEventListener('input', autoResizeTextarea);

userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        if (e.shiftKey) {
            // Let the new line be added naturally
            return;
        } else {
            // Prevent the default enter behavior and send message
            e.preventDefault();
            handleSubmit();
        }
    }
});

function switchMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.getElementById('chat-mode').classList.toggle('active', mode === 'chat');
    document.getElementById('search-mode').classList.toggle('active', mode === 'search');
    
    // Update placeholder text
    userInput.placeholder = mode === 'chat' 
        ? "Type your message here..." 
        : "Enter your search query...";
    
    // Focus the input
    userInput.focus();
}

function createTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        indicator.appendChild(dot);
    }
    return indicator;
}

function appendMessage(message, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

async function handleSubmit() {
    const message = userInput.value.trim();
    if (!message) return;

    // Disable input while processing
    userInput.disabled = true;
    
    // Add user message
    appendMessage(message, true);
    userInput.value = '';
    autoResizeTextarea();

    try {
        // Add typing indicator
        const typingIndicator = createTypingIndicator();
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const endpoint = currentMode === 'chat' ? '/chat' : '/search';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                query: message, // Send as both for compatibility
                stream: true 
            }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // Remove typing indicator and create response message div
        typingIndicator.remove();
        const responseDiv = appendMessage('', false);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            buffer += text;

            // Process complete SSE messages
            while (buffer.includes('\n\n')) {
                const [message, ...remaining] = buffer.split('\n\n');
                buffer = remaining.join('\n\n');

                if (message.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(message.slice(6));
                        if (data.chunk) {
                            // Add the new chunk character by character with a small delay
                            const chars = data.chunk.split('');
                            for (const char of chars) {
                                responseDiv.textContent += char;
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                                // Add a tiny delay between characters
                                await new Promise(resolve => setTimeout(resolve, 20));
                            }
                        }
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        appendMessage('Sorry, there was an error processing your request. Please try again.', false);
    } finally {
        // Re-enable input after processing
        userInput.disabled = false;
        userInput.focus();
    }
}
