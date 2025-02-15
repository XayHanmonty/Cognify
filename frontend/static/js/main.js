const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');

userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function appendMessage(message, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Disable input while processing
    userInput.disabled = true;
    appendMessage(message, true);
    userInput.value = '';

    try {
        // Create a message div for the response
        const responseDiv = appendMessage('', false);
        let fullResponse = '';

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message, stream: true }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.chunk) {
                            fullResponse += data.chunk;
                            responseDiv.textContent = fullResponse;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
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
