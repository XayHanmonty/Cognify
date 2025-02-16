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
            })
            
            // Format code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                return `<div class="code-block">
                    <div class="code-header">
                        ${lang ? `<span class="code-language">${lang}</span>` : ''}
                    </div>
                    <pre><code class="${lang || ''}">${code}</code></pre>
                </div>`;
            })
            // Handle paragraphs
            .split('\n\n')
            .map(para => {
                if (para.trim().startsWith('##')) {
                    // Handle headers
                    return `<h2>${para.replace('##', '').trim()}</h2>`;
                }
                if (para.trim().startsWith('**Key Points:**')) {
                    // Handle key points section
                    return `<div class="key-points">
                        <h3>${para.split('\n')[0]}</h3>
                        ${para.split('\n').slice(1).join('\n')}
                    </div>`;
                }
                if (para.trim().startsWith('Sources:')) {
                    // Handle sources section
                    return `<div class="sources">
                        <h3>Sources</h3>
                        ${para.replace('Sources:', '').trim()}
                    </div>`;
                }
                // Regular paragraphs
                return `<p>${para}</p>`;
            })
            .join('\n');

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

        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
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
                        if (data.response) {
                            // Add the response text character by character with a small delay
                            const chars = data.response.split('');
                            for (const char of chars) {
                                responseDiv.textContent += char;
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                                // Add a tiny delay between characters
                                await new Promise(resolve => setTimeout(resolve, 20));
                            }
                        } else if (data.error) {
                            responseDiv.textContent = `Error: ${data.error}`;
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
