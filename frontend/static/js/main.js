const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
let currentMode = 'chat'; 

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
    
    if (isUser) {
        messageDiv.textContent = message;
    } else {
        // For bot messages, handle formatting
        let formattedMessage = message
            // Format temperatures (e.g., 56°F)
            .replace(/(\d+)°([FC])/g, '<span class="temp">$1°$2</span>')
            
            // Format weather conditions
            .replace(/(Conditions: )(.*?)(?=\s*[•\n]|$)/g, '$1<span class="conditions">$2</span>')
            
            // Format date/time
            .replace(/((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})/g, 
                '<span class="datetime">$1</span>')
            
            // Handle bullet points while preserving spacing
            .replace(/^[•\-] (.*)/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
            
            // Handle Sources section
            .replace(/Sources:\n((?:\d+\. .*\n?)+)/g, (match, list) => {
                const items = list.split('\n')
                    .filter(item => item.trim())
                    .map(item => {
                        const [num, sourceFull] = item.split('. ');
                        const [name, url] = sourceFull.split(' - ').map(s => s.trim());
                        return `<li class="source-item">
                            <span class="source-number">${num}</span>
                            <a href="${url || '#'}" 
                               class="source-link" 
                               target="_blank"
                               rel="noopener noreferrer">
                                ${name}
                            </a>
                        </li>`;
                    })
                    .join('');
                return `<div class="sources" id="sources">
                    <h3>Sources</h3>
                    <ol>${items}</ol>
                </div>`;
            });
        
        // Handle paragraphs
        formattedMessage = formattedMessage
            .split('\n\n')
            .map(p => p.trim())
            .filter(p => p && !p.startsWith('<'))
            .map(p => `<p>${p}</p>`)
            .join('');
        
        messageDiv.innerHTML = formattedMessage;
    }
    
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
                query: message, 
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
