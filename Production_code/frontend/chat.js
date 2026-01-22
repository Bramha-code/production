// Auth
const userId = localStorage.getItem('user_id');
if (!userId) window.location.href = '/';

let currentSessionId = null;
let websocket = null;
let sessionWebSockets = {};
let deleteConfirmSessionId = null;
let renameSessionId = null;
let uploadedFiles = [];
let isCreatingNewChat = false;
let newChatPending = false;

// Streaming control
let isStreaming = false;
let currentStreamMessageId = null;
let interruptRequested = false;
let lastUserMessageContent = null;

// Theme
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);

document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    setupEventListeners();
    updateThemeIcon();
    updateSendButton(); // Initial button state
});

function setupEventListeners() {
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) newChatBtn.addEventListener('click', createNewChat);
    
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    
    const userInput = document.getElementById('user-input');
    if (userInput) {
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (isStreaming) {
                    interruptStream();
                } else {
                    sendMessage();
                }
            }
        });
        userInput.addEventListener('input', (e) => {
            autoResize(e.target);
            updateSendButton();
        });
    }
    
    // Enter key for rename modal
    const renameInput = document.getElementById('rename-input');
    if (renameInput) {
        renameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                confirmRenameChat();
            }
        });
    }
    
    // Enter key for profile modal
    const profileEmail = document.getElementById('profile-email');
    const profileFullname = document.getElementById('profile-fullname');
    if (profileEmail) {
        profileEmail.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveProfile();
            }
        });
    }
    if (profileFullname) {
        profileFullname.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveProfile();
            }
        });
    }
    
    document.querySelectorAll('.quick-prompt').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const prompt = e.currentTarget.dataset.prompt;
            if (userInput) {
                userInput.value = prompt;
                updateSendButton();
                sendMessage();
            }
        });
    });
    
    const attachBtn = document.getElementById('attach-btn');
    if (attachBtn) {
        attachBtn.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
    }
    
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.addEventListener('change', handleFileUpload);
    
    const sidebarToggle = document.getElementById('sidebar-toggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('collapsed');
        });
    }
    
    const searchInput = document.getElementById('search-input');
    if (searchInput) searchInput.addEventListener('input', handleSearch);
    
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
    
    const profileBtn = document.getElementById('profile-btn');
    if (profileBtn) profileBtn.addEventListener('click', showProfile);
    
    const faqBtn = document.getElementById('faq-btn');
    if (faqBtn) faqBtn.addEventListener('click', showFAQs);
    
    const saveProfileBtn = document.getElementById('save-profile-btn');
    if (saveProfileBtn) saveProfileBtn.addEventListener('click', saveProfile);
    
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) logoutBtn.addEventListener('click', logout);

    // PDF Export button in header
    const exportPdfBtn = document.getElementById('export-pdf-btn');
    if (exportPdfBtn) exportPdfBtn.addEventListener('click', showExportPdfModal);

    // Generate PDF button in modal
    const generatePdfBtn = document.getElementById('generate-pdf-btn');
    if (generatePdfBtn) generatePdfBtn.addEventListener('click', generateAndDownloadPdf);

    document.querySelectorAll('.close-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        });
    });
    
    const confirmDelete = document.getElementById('confirm-delete');
    if (confirmDelete) confirmDelete.addEventListener('click', confirmDeleteChat);
    
    const cancelDelete = document.getElementById('cancel-delete');
    if (cancelDelete) {
        cancelDelete.addEventListener('click', () => {
            document.getElementById('delete-confirm').classList.remove('active');
        });
    }
    
    const confirmRename = document.getElementById('confirm-rename');
    if (confirmRename) confirmRename.addEventListener('click', confirmRenameChat);
    
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.chat-item-menu') && !e.target.closest('.menu-dropdown')) {
            document.querySelectorAll('.menu-dropdown').forEach(d => d.remove());
        }
    });
}

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const newTheme = current === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.querySelector('#theme-toggle i');
    if (icon) {
        const theme = document.documentElement.getAttribute('data-theme');
        icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function updateSendButton() {
    const input = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const sendIcon = sendBtn?.querySelector('i');
    
    if (!sendBtn || !sendIcon) return;
    
    if (isStreaming) {
        // Show pause button during streaming
        sendBtn.className = 'pause-btn';
        sendBtn.title = 'Stop generation';
        sendBtn.disabled = false;
        sendIcon.className = 'fas fa-pause';
    } else {
        // Show send button when not streaming
        sendBtn.className = 'send-btn';
        sendBtn.title = 'Send message';
        sendBtn.disabled = !input?.value.trim() && uploadedFiles.length === 0;
        sendIcon.className = 'fas fa-arrow-up';
    }
}

function setStreamingState(streaming, messageId = null) {
    isStreaming = streaming;
    currentStreamMessageId = messageId;
    updateSendButton();
}

async function interruptStream() {
    if (!isStreaming || !currentSessionId) return;
    
    console.log('🛑 Interrupting stream...');
    interruptRequested = true;
    
    // Send interrupt signal via WebSocket
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ interrupt: true }));
        console.log('Interrupt signal sent');
    }
    
    // Immediately stop streaming state on client side
    setStreamingState(false);
    
    // Remove the assistant message completely
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        const currentMsg = document.querySelector(`[data-message-id="${currentStreamMessageId}"]`);
        if (currentMsg) {
            // Remove the entire assistant message
            currentMsg.remove();
            console.log('Assistant message removed from UI');
        }
        
        // Also remove the last user message if it matches the interrupted one
        const lastUserMsg = chatMessages.querySelector('.message.user:last-child');
        if (lastUserMsg && lastUserMessageContent) {
            const msgText = lastUserMsg.querySelector('.message-text');
            if (msgText && msgText.textContent === lastUserMessageContent) {
                lastUserMsg.remove();
                console.log('User message also removed from UI');
            }
        }
    }
    
    // Clear the last user message content
    lastUserMessageContent = null;
    console.log('Stream fully interrupted - nothing saved or displayed');
}

async function loadSessions() {
    try {
        const response = await fetch(`/api/sessions/${userId}`);
        const data = await response.json();
        renderSessions(data.sessions);
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

function renderSessions(sessions) {
    const list = document.getElementById('chat-list');
    if (!list) return;
    list.innerHTML = '';
    
    const starred = sessions.filter(s => s.starred);
    const recent = sessions.filter(s => !s.starred);
    
    // Starred section
    if (starred.length > 0) {
        const starredHeader = document.createElement('div');
        starredHeader.className = 'chat-section-header';
        starredHeader.textContent = 'Starred';
        list.appendChild(starredHeader);
        
        starred.forEach(session => {
            list.appendChild(createChatItem(session));
        });
    }
    
    // Recent section
    if (recent.length > 0) {
        const recentHeader = document.createElement('div');
        recentHeader.className = 'chat-section-header';
        recentHeader.textContent = 'Recent';
        list.appendChild(recentHeader);
        
        recent.forEach(session => {
            list.appendChild(createChatItem(session));
        });
    }
}

function createChatItem(session) {
    const item = document.createElement('div');
    item.className = 'chat-item';
    if (session.session_id === currentSessionId) {
        item.classList.add('active');
    }
    
    item.innerHTML = `
        <div class="chat-item-content">
            <div class="chat-item-title">${escapeHtml(session.title)}</div>
            <div class="chat-item-time">${formatTime(session.updated_at)}</div>
        </div>
        <button class="menu-btn more-btn" data-id="${session.session_id}">
            <i class="fas fa-ellipsis-v"></i>
        </button>
    `;
    
    const contentDiv = item.querySelector('.chat-item-content');
    contentDiv.addEventListener('click', () => {
        loadSession(session.session_id);
    });
    
    const moreBtn = item.querySelector('.more-btn');
    moreBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showChatMenu(e, session);
    });
    
    return item;
}

function showChatMenu(event, session) {
    document.querySelectorAll('.menu-dropdown').forEach(d => d.remove());
    
    const dropdown = document.createElement('div');
    dropdown.className = 'menu-dropdown';
    dropdown.innerHTML = `
        <button class="dropdown-item star-item">
            <i class="fas fa-star ${session.starred ? 'starred' : ''}"></i>
            ${session.starred ? 'Unstar' : 'Star'}
        </button>
        <button class="dropdown-item rename-item">
            <i class="fas fa-edit"></i>
            Rename
        </button>
        <button class="dropdown-item delete-item">
            <i class="fas fa-trash"></i>
            Delete
        </button>
    `;
    
    const rect = event.target.closest('.menu-btn').getBoundingClientRect();
    dropdown.style.position = 'absolute';
    dropdown.style.left = (rect.left - 120) + 'px';
    dropdown.style.top = (rect.bottom + 5) + 'px';
    dropdown.style.zIndex = '1000';
    
    document.body.appendChild(dropdown);
    
    dropdown.querySelector('.star-item').addEventListener('click', async () => {
        await starSession(session.session_id);
        dropdown.remove();
    });
    
    dropdown.querySelector('.rename-item').addEventListener('click', () => {
        showRenameDialog(session.session_id, session.title);
        dropdown.remove();
    });
    
    dropdown.querySelector('.delete-item').addEventListener('click', () => {
        showDeleteConfirm(session.session_id);
        dropdown.remove();
    });
}

async function createNewChat() {
    if (isCreatingNewChat || newChatPending) return;
    
    try {
        isCreatingNewChat = true;
        newChatPending = true;
        
        const response = await fetch(`/api/sessions/${userId}`, { method: 'POST' });
        const data = await response.json();
        currentSessionId = data.session_id;
        
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) chatMessages.innerHTML = '';
        
        const welcomeScreen = document.getElementById('welcome-screen');
        if (welcomeScreen) welcomeScreen.style.display = 'flex';
        
        const messagesContainer = document.getElementById('messages-container');
        if (messagesContainer) messagesContainer.classList.add('centered');
        
        const chatTitle = document.getElementById('current-chat-title');
        if (chatTitle) chatTitle.textContent = 'New Chat';
        
        connectWebSocket();
        await loadSessions();
    } catch (error) {
        console.error('Error creating session:', error);
    } finally {
        isCreatingNewChat = false;
    }
}

async function loadSession(sessionId) {
    try {
        const response = await fetch(`/api/session/${sessionId}`);
        const session = await response.json();
        
        currentSessionId = sessionId;
        newChatPending = false;
        
        const welcomeScreen = document.getElementById('welcome-screen');
        if (welcomeScreen) welcomeScreen.style.display = 'none';
        
        const messagesContainer = document.getElementById('messages-container');
        if (messagesContainer) messagesContainer.classList.remove('centered');
        
        const chatTitle = document.getElementById('current-chat-title');
        if (chatTitle) chatTitle.textContent = session.title;
        
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
            session.messages.forEach(msg => displayMessage(msg));
            
            // Multiple scroll attempts to ensure it works
            const scrollToBottom = () => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
                const lastMessage = chatMessages.lastElementChild;
                if (lastMessage) {
                    lastMessage.scrollIntoView({ behavior: 'instant', block: 'end' });
                }
            };
            
            // Immediate scroll
            scrollToBottom();
            
            // Delayed scroll after rendering
            setTimeout(scrollToBottom, 50);
            setTimeout(scrollToBottom, 150);
            setTimeout(scrollToBottom, 300);
        }
        
        connectWebSocket();
        await loadSessions();
    } catch (error) {
        console.error('Error loading session:', error);
    }
}

async function starSession(sessionId) {
    try {
        await fetch(`/api/session/${sessionId}/star`, { method: 'POST' });
        await loadSessions();
    } catch (error) {
        console.error('Error starring session:', error);
    }
}

function showDeleteConfirm(sessionId) {
    deleteConfirmSessionId = sessionId;
    const deleteConfirm = document.getElementById('delete-confirm');
    if (deleteConfirm) {
        deleteConfirm.classList.add('active');
        // Position near sidebar
        deleteConfirm.style.position = 'fixed';
        deleteConfirm.style.left = '350px';
        deleteConfirm.style.top = '50%';
        deleteConfirm.style.transform = 'translateY(-50%)';
    }
}

async function confirmDeleteChat() {
    if (!deleteConfirmSessionId) return;
    
    try {
        await fetch(`/api/session/${deleteConfirmSessionId}`, { method: 'DELETE' });
        
        if (deleteConfirmSessionId === currentSessionId) {
            currentSessionId = null;
            newChatPending = false;
            
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) chatMessages.innerHTML = '';
            
            const welcomeScreen = document.getElementById('welcome-screen');
            if (welcomeScreen) welcomeScreen.style.display = 'flex';
            
            const messagesContainer = document.getElementById('messages-container');
            if (messagesContainer) messagesContainer.classList.add('centered');
            
            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) chatTitle.textContent = 'New Chat';
        }
        
        const deleteConfirm = document.getElementById('delete-confirm');
        if (deleteConfirm) deleteConfirm.classList.remove('active');
        
        await loadSessions();
    } catch (error) {
        console.error('Error deleting session:', error);
    }
}

function showRenameDialog(sessionId, currentTitle) {
    renameSessionId = sessionId;
    const renameInput = document.getElementById('rename-input');
    if (renameInput) {
        renameInput.value = currentTitle;
        renameInput.focus();
        renameInput.select();
    }
    
    const renameModal = document.getElementById('rename-modal');
    if (renameModal) renameModal.classList.add('active');
}

async function confirmRenameChat() {
    const renameInput = document.getElementById('rename-input');
    const newTitle = renameInput ? renameInput.value.trim() : '';
    if (!newTitle || !renameSessionId) return;
    
    try {
        await fetch(`/api/session/${renameSessionId}/rename`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_title: newTitle })
        });
        
        if (renameSessionId === currentSessionId) {
            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) chatTitle.textContent = newTitle;
        }
        
        const renameModal = document.getElementById('rename-modal');
        if (renameModal) renameModal.classList.remove('active');
        
        await loadSessions();
    } catch (error) {
        console.error('Error renaming session:', error);
    }
}

async function handleSearch(e) {
    const query = e.target.value.trim();
    
    if (!query) {
        loadSessions();
        return;
    }
    
    try {
        const response = await fetch(`/api/search/${userId}?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        renderSessions(data.sessions);
    } catch (error) {
        console.error('Error searching:', error);
    }
}

function connectWebSocket() {
    if (!currentSessionId) {
        console.log('No session ID, skipping WebSocket connection');
        return;
    }
    
    if (sessionWebSockets[currentSessionId]) {
        websocket = sessionWebSockets[currentSessionId];
        console.log('Reusing existing WebSocket for session:', currentSessionId);
        return;
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${currentSessionId}`;
    console.log('Connecting WebSocket to:', wsUrl);
    
    websocket = new WebSocket(wsUrl);
    sessionWebSockets[currentSessionId] = websocket;
    
    websocket.onopen = () => {
        console.log('WebSocket connected successfully');
        interruptRequested = false;
        lastUserMessageContent = null;
    };
    
    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            // Check if interrupt was requested - ignore ALL incoming messages
            if (interruptRequested) {
                console.log('Ignoring ALL WebSocket messages after interrupt:', data.type);
                return;
            }
            
            console.log('WebSocket message received:', data.type);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error, event.data);
        }
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStreamingState(false);
        interruptRequested = false;
    };
    
    websocket.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        delete sessionWebSockets[currentSessionId];
        setStreamingState(false);
        interruptRequested = false;
        lastUserMessageContent = null;
    };
}

function handleWebSocketMessage(data) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error('chat-messages container not found');
        return;
    }
    
    if (data.type === 'metadata') {
        console.log('Creating assistant message container for:', data.data.message_id);
        setStreamingState(true, data.data.message_id);
        
        // Create assistant message element
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'message assistant';
        assistantMsg.setAttribute('data-message-id', data.data.message_id);
        assistantMsg.setAttribute('id', `msg-${data.data.message_id}`);
        
        const msgContent = document.createElement('div');
        msgContent.className = 'message-content';
        
        const msgText = document.createElement('div');
        msgText.className = 'message-text';
        msgText.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
        msgText.dataset.rawText = '';
        
        msgContent.appendChild(msgText);
        assistantMsg.appendChild(msgContent);
        chatMessages.appendChild(assistantMsg);
        
        console.log('Assistant message container created');
        
    } else if (data.type === 'stream') {
        const messageId = data.data.message_id;
        let assistantMsg = document.querySelector(`[data-message-id="${messageId}"]`);
        
        if (!assistantMsg) {
            console.log('Message element not found, creating new one for:', messageId);
            assistantMsg = document.createElement('div');
            assistantMsg.className = 'message assistant';
            assistantMsg.setAttribute('data-message-id', messageId);
            assistantMsg.setAttribute('id', `msg-${messageId}`);
            
            const msgContent = document.createElement('div');
            msgContent.className = 'message-content';
            
            const msgText = document.createElement('div');
            msgText.className = 'message-text';
            msgText.dataset.rawText = '';
            
            msgContent.appendChild(msgText);
            assistantMsg.appendChild(msgContent);
            chatMessages.appendChild(assistantMsg);
        }
        
        const msgText = assistantMsg.querySelector('.message-text');
        if (!msgText) {
            console.error('Message text element not found');
            return;
        }
        
        // Remove typing indicator if present
        if (msgText.querySelector('.typing-indicator')) {
            msgText.innerHTML = '';
            msgText.dataset.rawText = '';
        }
        
        // Append new content
        if (!msgText.dataset.rawText) msgText.dataset.rawText = '';
        msgText.dataset.rawText += data.data.content;
        
        // Update display with formatted markdown
        msgText.innerHTML = formatMarkdown(msgText.dataset.rawText);
        
        // Scroll to show new content
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);
        
    } else if (data.type === 'complete') {
        console.log('Stream complete for message:', data.data.message_id);
        const messageId = data.data.message_id;
        const assistantMsg = document.querySelector(`[data-message-id="${messageId}"]`);
        
        if (assistantMsg) {
            const msgText = assistantMsg.querySelector('.message-text');
            if (msgText && msgText.dataset.rawText) {
                // Final formatting
                msgText.innerHTML = formatMarkdown(msgText.dataset.rawText);
                delete msgText.dataset.rawText;
            }
        }
        
        // Update session list
        newChatPending = false;
        loadSessions();
        
        // Final scroll
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
        
        setStreamingState(false);
        interruptRequested = false;
        lastUserMessageContent = null;
        
    } else if (data.type === 'error') {
        console.error('WebSocket error:', data.data);
        const lastMsg = chatMessages.querySelector('.message.assistant:last-child .message-text');
        if (lastMsg) {
            lastMsg.innerHTML = '<p style="color: #ef4444;">Error: ' + escapeHtml(data.data.message || 'Unknown error') + '</p>';
        }
        setStreamingState(false);
        interruptRequested = false;
        lastUserMessageContent = null;
    }
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input ? input.value.trim() : '';
    
    if (!message && uploadedFiles.length === 0) return;
    
    // Store the message content for potential removal if interrupted
    lastUserMessageContent = message;
    
    if (!currentSessionId || newChatPending) {
        await createNewChat();
        await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    const isFirstMessage = document.getElementById('chat-messages')?.children.length === 0;
    if (isFirstMessage && message) {
        const newTitle = message.substring(0, 50).trim();
        try {
            await fetch(`/api/session/${currentSessionId}/rename`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_title: newTitle })
            });
            
            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) chatTitle.textContent = newTitle;
            
            loadSessions();
        } catch (error) {
            console.error('Error renaming chat:', error);
        }
    }
    
    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    
    const messagesContainer = document.getElementById('messages-container');
    if (messagesContainer) messagesContainer.classList.remove('centered');
    
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        const userMsg = createMessageElement('user', message);
        
        if (uploadedFiles.length > 0) {
            const attachmentsDiv = document.createElement('div');
            attachmentsDiv.className = 'message-attachments';
            
            uploadedFiles.forEach(file => {
                const attachment = document.createElement('div');
                const isImage = file.filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);
                
                if (isImage) {
                    attachment.className = 'attachment-image';
                    attachment.innerHTML = `<img src="/api/file/${userId}/${file.file_id}" alt="${escapeHtml(file.filename)}">`;
                } else {
                    attachment.className = 'attachment-doc';
                    attachment.innerHTML = `
                        <i class="fas fa-file-alt"></i>
                        <span>${escapeHtml(file.filename)}</span>
                    `;
                }
                
                attachmentsDiv.appendChild(attachment);
            });
            
            const msgContent = userMsg.querySelector('.message-content');
            msgContent.insertBefore(attachmentsDiv, msgContent.firstChild);
        }
        
        chatMessages.appendChild(userMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    if (input) {
        input.value = '';
        autoResize(input);
    }
    updateSendButton();
    
    const filesData = [...uploadedFiles];
    uploadedFiles = [];
    const filePreview = document.getElementById('file-preview');
    if (filePreview) filePreview.innerHTML = '';
    
    const messageWithFiles = {
        message: message,
        files: filesData.map(f => ({ 
            filename: f.filename, 
            file_id: f.file_id,
            temp_path: f.path
        }))
    };
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(messageWithFiles));
        console.log('Message sent via WebSocket');
    } else {
        console.log('WebSocket not open, connecting...');
        connectWebSocket();
        setTimeout(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify(messageWithFiles));
                console.log('Message sent after reconnection');
            } else {
                console.error('WebSocket still not open');
            }
        }, 1000);
    }
}

function createMessageElement(role, content) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    
    const msgContent = document.createElement('div');
    msgContent.className = 'message-content';
    
    if (content) {
        const msgText = document.createElement('div');
        msgText.className = 'message-text';
        msgText.textContent = content;
        msgContent.appendChild(msgText);
    }
    
    msg.appendChild(msgContent);
    
    return msg;
}

function renderGraphVisualization(container, graphData) {
    if (!container || typeof vis === 'undefined') {
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #ef4444; padding: 2rem;">Graph library not loaded</p>';
        }
        return;
    }
    
    try {
        const nodes = new vis.DataSet(graphData.nodes.map(node => ({
            id: node.id,
            label: node.label,
            title: node.title,
            color: node.type === 'main' ? '#ef4444' : '#f59e0b',
            size: node.type === 'main' ? 30 : 20,
            font: { size: 12, color: '#1f2937' },
            fixed: { x: false, y: false }
        })));
        
        const edges = new vis.DataSet(graphData.edges.map(edge => ({
            from: edge.from,
            to: edge.to,
            arrows: 'to',
            color: edge.type === 'hierarchical' ? '#3b82f6' : edge.type === 'semantic' ? '#10b981' : '#a855f7',
            smooth: { type: 'continuous' }
        })));
        
        const network = new vis.Network(container, { nodes, edges }, {
            physics: { 
                enabled: true,
                stabilization: { 
                    enabled: true,
                    iterations: 200,
                    fit: true
                },
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                }
            },
            interaction: { 
                hover: true, 
                tooltipDelay: 100,
                dragNodes: false,
                dragView: true,
                zoomView: true
            }
        });
        
        // Stop physics after stabilization
        network.on('stabilizationIterationsDone', function() {
            network.setOptions({ physics: false });
        });
        
        // Fallback: stop after 3 seconds
        setTimeout(() => {
            network.setOptions({ physics: false });
        }, 3000);
        
    } catch (error) {
        console.error('Error rendering graph:', error);
        container.innerHTML = '<p style="text-align: center; color: #ef4444; padding: 2rem;">Error rendering graph</p>';
    }
}

function displayMessage(msg) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const msgEl = createMessageElement(msg.role, msg.content);
    
    // Display attachments from session storage
    if (msg.attachments && msg.attachments.length > 0) {
        const attachmentsDiv = document.createElement('div');
        attachmentsDiv.className = 'message-attachments';
        
        msg.attachments.forEach(att => {
            const attachment = document.createElement('div');
            const isImage = att.filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);
            
            if (isImage) {
                attachment.className = 'attachment-image';
                attachment.innerHTML = `<img src="/api/attachment/${att.path}" alt="${escapeHtml(att.filename)}">`;
            } else {
                attachment.className = 'attachment-doc';
                attachment.innerHTML = `
                    <i class="fas fa-file-alt"></i>
                    <span>${escapeHtml(att.filename)}</span>
                `;
            }
            
            attachmentsDiv.appendChild(attachment);
        });
        
        const msgContent = msgEl.querySelector('.message-content');
        msgContent.insertBefore(attachmentsDiv, msgContent.firstChild);
    }
    // Fallback to old images field for backward compatibility
    else if (msg.role === 'user' && msg.images && msg.images.length > 0) {
        const attachmentsDiv = document.createElement('div');
        attachmentsDiv.className = 'message-attachments';
        
        msg.images.forEach(filename => {
            const attachment = document.createElement('div');
            const isImage = filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);
            
            if (isImage) {
                attachment.className = 'attachment-image';
                attachment.innerHTML = `<img src="/uploads/${filename}" alt="${escapeHtml(filename)}">`;
            } else {
                attachment.className = 'attachment-doc';
                attachment.innerHTML = `
                    <i class="fas fa-file-alt"></i>
                    <span>${escapeHtml(filename)}</span>
                `;
            }
            
            attachmentsDiv.appendChild(attachment);
        });
        
        const msgContent = msgEl.querySelector('.message-content');
        msgContent.insertBefore(attachmentsDiv, msgContent.firstChild);
    }
    
    if (msg.role === 'assistant') {
        const msgText = msgEl.querySelector('.message-text');
        msgText.innerHTML = formatMarkdown(msg.content);
        
        if (msg.graph_data && msg.graph_data.nodes && msg.graph_data.nodes.length > 0) {
            const stats = {
                nodes: msg.graph_data.nodes.length,
                connected: msg.graph_data.nodes.filter(n => n.type === 'connected').length,
                images: msg.images ? msg.images.length : 0
            };
            displayGraph(msg.graph_data, stats);
        }
        
        if (msg.images && msg.images.length > 0) {
            displayImages(msg.images);
        }
    }
    
    chatMessages.appendChild(msgEl);
}

function displayGraph(graphData, stats) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    // Store graph reference to prevent removal
    const graphDiv = document.createElement('div');
    graphDiv.className = 'graph-visualization';
    graphDiv.setAttribute('data-persistent', 'true');
    const graphId = 'graph-' + Date.now();
    graphDiv.innerHTML = `
        <div class="graph-header">
            <h3><i class="fas fa-project-diagram"></i> Knowledge Graph</h3>
            <div class="graph-stats">
                <span><i class="fas fa-circle-nodes"></i> ${stats.nodes} nodes</span>
                <span><i class="fas fa-link"></i> ${stats.connected} connected</span>
                <span><i class="fas fa-image"></i> ${stats.images} images</span>
            </div>
        </div>
        <div class="graph-network" id="${graphId}"></div>
        <div class="graph-legend">
            <div class="legend-item"><span class="legend-dot main"></span> Main Node</div>
            <div class="legend-item"><span class="legend-dot connected"></span> Connected Nodes</div>
        </div>
    `;
    
    chatMessages.appendChild(graphDiv);
    
    // Render graph using the function
    setTimeout(() => {
        const container = document.getElementById(graphId);
        if (container) {
            renderGraphVisualization(container, graphData);
        }
    }, 100);
}

function displayImages(imagePaths) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const galleryDiv = document.createElement('div');
    galleryDiv.className = 'image-gallery';
    galleryDiv.innerHTML = `
        <div class="gallery-header">
            <i class="fas fa-images"></i> Retrieved Images (${imagePaths.length})
        </div>
        <div class="gallery-grid">
            ${imagePaths.map(path => `
                <div class="gallery-item">
                    <img src="/api/image?path=${encodeURIComponent(path)}" alt="Image" loading="lazy">
                </div>
            `).join('')}
        </div>
    `;
    
    chatMessages.appendChild(galleryDiv);
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`/api/upload/${userId}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        uploadedFiles.push(data);
        
        const preview = document.getElementById('file-preview');
        if (preview) {
            const isImage = file.type.startsWith('image/');
            const chip = document.createElement('div');
            chip.className = 'file-attachment';
            
            if (isImage) {
                // Create image preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    chip.innerHTML = `
                        <img src="${e.target.result}" alt="${escapeHtml(file.name)}">
                        <button class="remove-attachment" onclick="removeFile('${data.file_id}')">
                            <i class="fas fa-times"></i>
                        </button>
                    `;
                };
                reader.readAsDataURL(file);
            } else {
                // Create document icon
                chip.innerHTML = `
                    <div class="doc-attachment">
                        <i class="fas fa-file-alt"></i>
                        <span class="doc-name">${escapeHtml(file.name)}</span>
                    </div>
                    <button class="remove-attachment" onclick="removeFile('${data.file_id}')">
                        <i class="fas fa-times"></i>
                    </button>
                `;
            }
            
            preview.appendChild(chip);
        }
        
        updateSendButton();
        e.target.value = ''; // Clear file input
    } catch (error) {
        console.error('Error uploading file:', error);
    }
}

function removeFile(fileId) {
    uploadedFiles = uploadedFiles.filter(f => f.file_id !== fileId);
    const preview = document.getElementById('file-preview');
    if (preview) {
        Array.from(preview.children).forEach(chip => {
            if (chip.innerHTML.includes(fileId)) chip.remove();
        });
    }
    updateSendButton();
}

async function showProfile() {
    try {
        const response = await fetch(`/api/profile/${userId}`);
        const data = await response.json();
        
        const username = document.getElementById('profile-username');
        if (username) username.value = data.username;
        
        const email = document.getElementById('profile-email');
        if (email) email.value = data.email;
        
        const fullname = document.getElementById('profile-fullname');
        if (fullname) fullname.value = data.full_name || '';
        
        const modal = document.getElementById('profile-modal');
        if (modal) modal.classList.add('active');
    } catch (error) {
        console.error('Error loading profile:', error);
    }
}

async function saveProfile() {
    const emailEl = document.getElementById('profile-email');
    const fullNameEl = document.getElementById('profile-fullname');
    const email = emailEl ? emailEl.value : '';
    const fullName = fullNameEl ? fullNameEl.value : '';
    
    try {
        await fetch(`/api/profile/${userId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, full_name: fullName })
        });
        
        alert('Profile updated');
    } catch (error) {
        console.error('Error saving profile:', error);
    }
}

function logout() {
    localStorage.clear();
    window.location.href = '/';
}

async function showFAQs() {
    try {
        const response = await fetch('/api/faqs');
        const data = await response.json();
        
        const content = document.getElementById('faq-content');
        if (content) {
            content.innerHTML = data.faqs.map(faq => `
                <div class="faq-item" onclick="this.classList.toggle('active')">
                    <div class="faq-question">
                        ${escapeHtml(faq.question)}
                        <i class="fas fa-chevron-down"></i>
                    </div>
                    <div class="faq-answer">${escapeHtml(faq.answer)}</div>
                </div>
            `).join('');
        }
        
        const modal = document.getElementById('faq-modal');
        if (modal) modal.classList.add('active');
    } catch (error) {
        console.error('Error loading FAQs:', error);
    }
}

function formatMarkdown(text) {
    let html = escapeHtml(text);
    
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    const lines = html.split('\n');
    let inTable = false;
    let tableRows = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        if (line.startsWith('|') && line.endsWith('|')) {
            const cells = line.split('|').filter(c => c.trim()).map(c => c.trim());
            if (cells.every(c => /^[-:]+$/.test(c))) continue;
            
            if (!inTable) {
                inTable = true;
                tableRows = [];
            }
            tableRows.push(cells);
        } else if (inTable) {
            if (tableRows.length > 0) {
                let table = '<table>';
                tableRows.forEach((row, idx) => {
                    table += '<tr>';
                    const tag = idx === 0 ? 'th' : 'td';
                    row.forEach(cell => {
                        table += `<${tag}>${cell}</${tag}>`;
                    });
                    table += '</tr>';
                });
                table += '</table>';
                
                lines[i - tableRows.length - 1] = table;
                for (let j = i - tableRows.length; j < i; j++) {
                    lines[j] = '';
                }
            }
            inTable = false;
            tableRows = [];
        }
    }
    
    html = lines.join('\n');
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*?<\/li>\n?)+/g, match => `<ul>${match}</ul>`);
    
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p>(<h[1-4]>)/g, '$1');
    html = html.replace(/(<\/h[1-4]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<table>)/g, '$1');
    html = html.replace(/(<\/table>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');
    
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;

    return date.toLocaleDateString();
}

// =========================================================
// Test Plan Export Functions
// =========================================================

// Track the last test plan query for export
let lastTestPlanQuery = null;

function showExportPdfModal() {
    const modal = document.getElementById('export-pdf-modal');
    const queryInput = document.getElementById('pdf-query-input');
    const statusDiv = document.getElementById('pdf-status');

    // Clear previous status
    if (statusDiv) statusDiv.innerHTML = '';

    // Pre-fill with a default query
    if (queryInput && !queryInput.value) {
        queryInput.value = 'EMC test plan for automotive electronics';
    }

    if (modal) modal.classList.add('active');
}

async function generateAndDownloadPdf() {
    const queryInput = document.getElementById('pdf-query-input');
    const includeRecs = document.getElementById('include-recommendations');
    const statusDiv = document.getElementById('pdf-status');
    const generateBtn = document.getElementById('generate-pdf-btn');

    const query = queryInput ? queryInput.value.trim() : '';
    if (!query) {
        if (statusDiv) {
            statusDiv.innerHTML = '<div class="status-error"><i class="fas fa-exclamation-circle"></i> Please enter a test plan query</div>';
        }
        return;
    }

    const originalBtnHtml = generateBtn.innerHTML;

    try {
        // Show loading state
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
        generateBtn.disabled = true;

        if (statusDiv) {
            statusDiv.innerHTML = '<div class="status-loading"><i class="fas fa-cog fa-spin"></i> Generating test plan and creating PDF... This may take a minute.</div>';
        }

        const response = await fetch('/api/v2/test-plan/export/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                include_recommendations: includeRecs ? includeRecs.checked : true
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate PDF');
        }

        // Get the PDF blob
        const blob = await response.blob();

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'EMC_TestPlan.pdf';
        if (contentDisposition) {
            const match = contentDisposition.match(/filename=(.+)/);
            if (match) filename = match[1];
        }

        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        // Show success
        if (statusDiv) {
            statusDiv.innerHTML = '<div class="status-success"><i class="fas fa-check-circle"></i> PDF downloaded successfully!</div>';
        }

        generateBtn.innerHTML = '<i class="fas fa-check"></i> Downloaded!';
        setTimeout(() => {
            generateBtn.innerHTML = originalBtnHtml;
            generateBtn.disabled = false;
        }, 2000);

    } catch (error) {
        console.error('PDF generation failed:', error);
        if (statusDiv) {
            statusDiv.innerHTML = `<div class="status-error"><i class="fas fa-exclamation-circle"></i> ${error.message}</div>`;
        }
        generateBtn.innerHTML = originalBtnHtml;
        generateBtn.disabled = false;
    }
}

function isTestPlanContent(content) {
    // Check if the content looks like a test plan response
    const testPlanIndicators = [
        'test plan',
        'test case',
        'Test Case',
        'TEST PLAN',
        'requirement',
        'procedure',
        'pass criteria',
        'EMC test'
    ];

    const lowerContent = content.toLowerCase();
    let matches = 0;
    for (const indicator of testPlanIndicators) {
        if (lowerContent.includes(indicator.toLowerCase())) {
            matches++;
        }
    }
    return matches >= 3;
}

function addExportButton(messageElement, query) {
    const msgContent = messageElement.querySelector('.message-content');
    if (!msgContent) return;

    // Check if export button already exists
    if (msgContent.querySelector('.export-actions')) return;

    const exportDiv = document.createElement('div');
    exportDiv.className = 'export-actions';
    exportDiv.innerHTML = `
        <div class="export-toolbar">
            <span class="export-label"><i class="fas fa-file-export"></i> Export Test Plan:</span>
            <button class="export-btn pdf-btn" onclick="exportTestPlanPDF('${escapeHtml(query.replace(/'/g, "\\'"))}')">
                <i class="fas fa-file-pdf"></i> Download PDF
            </button>
        </div>
    `;

    msgContent.appendChild(exportDiv);
}

async function exportTestPlanPDF(query) {
    const btn = event.target.closest('.export-btn');
    const originalHtml = btn.innerHTML;

    try {
        // Show loading state
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        btn.disabled = true;

        const response = await fetch('/api/v2/test-plan/export/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                include_recommendations: true
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate PDF');
        }

        // Get the PDF blob
        const blob = await response.blob();

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'TestPlan.pdf';
        if (contentDisposition) {
            const match = contentDisposition.match(/filename=(.+)/);
            if (match) filename = match[1];
        }

        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        // Show success
        btn.innerHTML = '<i class="fas fa-check"></i> Downloaded!';
        setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.disabled = false;
        }, 2000);

    } catch (error) {
        console.error('PDF export failed:', error);
        btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed';
        setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.disabled = false;
        }, 2000);
        alert('Failed to export PDF: ' + error.message);
    }
}

// Override or enhance handleWebSocketMessage to detect test plans
const originalHandleWebSocketMessage = handleWebSocketMessage;
handleWebSocketMessage = function(data) {
    // Call original handler
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error('chat-messages container not found');
        return;
    }

    if (data.type === 'metadata') {
        console.log('Creating assistant message container for:', data.data.message_id);
        setStreamingState(true, data.data.message_id);

        // Create assistant message element
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'message assistant';
        assistantMsg.setAttribute('data-message-id', data.data.message_id);
        assistantMsg.setAttribute('id', `msg-${data.data.message_id}`);

        const msgContent = document.createElement('div');
        msgContent.className = 'message-content';

        const msgText = document.createElement('div');
        msgText.className = 'message-text';
        msgText.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
        msgText.dataset.rawText = '';

        msgContent.appendChild(msgText);
        assistantMsg.appendChild(msgContent);
        chatMessages.appendChild(assistantMsg);

        console.log('Assistant message container created');

    } else if (data.type === 'stream') {
        const messageId = data.data.message_id;
        let assistantMsg = document.querySelector(`[data-message-id="${messageId}"]`);

        if (!assistantMsg) {
            console.log('Message element not found, creating new one for:', messageId);
            assistantMsg = document.createElement('div');
            assistantMsg.className = 'message assistant';
            assistantMsg.setAttribute('data-message-id', messageId);
            assistantMsg.setAttribute('id', `msg-${messageId}`);

            const msgContent = document.createElement('div');
            msgContent.className = 'message-content';

            const msgText = document.createElement('div');
            msgText.className = 'message-text';
            msgText.dataset.rawText = '';

            msgContent.appendChild(msgText);
            assistantMsg.appendChild(msgContent);
            chatMessages.appendChild(assistantMsg);
        }

        const msgText = assistantMsg.querySelector('.message-text');
        if (!msgText) {
            console.error('Message text element not found');
            return;
        }

        // Remove typing indicator if present
        if (msgText.querySelector('.typing-indicator')) {
            msgText.innerHTML = '';
            msgText.dataset.rawText = '';
        }

        // Append new content
        if (!msgText.dataset.rawText) msgText.dataset.rawText = '';
        msgText.dataset.rawText += data.data.content;

        // Update display with formatted markdown
        msgText.innerHTML = formatMarkdown(msgText.dataset.rawText);

        // Scroll to show new content
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);

    } else if (data.type === 'complete') {
        console.log('Stream complete for message:', data.data.message_id);
        const messageId = data.data.message_id;
        const assistantMsg = document.querySelector(`[data-message-id="${messageId}"]`);

        if (assistantMsg) {
            const msgText = assistantMsg.querySelector('.message-text');
            if (msgText && msgText.dataset.rawText) {
                // Final formatting
                msgText.innerHTML = formatMarkdown(msgText.dataset.rawText);

                // Check if this is a test plan response and add export button
                if (isTestPlanContent(msgText.dataset.rawText) && lastUserMessageContent) {
                    setTimeout(() => {
                        addExportButton(assistantMsg, lastUserMessageContent);
                    }, 100);
                }

                delete msgText.dataset.rawText;
            }
        }

        // Update session list
        newChatPending = false;
        loadSessions();

        // Final scroll
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);

        setStreamingState(false);
        interruptRequested = false;
        lastUserMessageContent = null;

    } else if (data.type === 'error') {
        console.error('WebSocket error:', data.data);
        const lastMsg = chatMessages.querySelector('.message.assistant:last-child .message-text');
        if (lastMsg) {
            lastMsg.innerHTML = '<p style="color: #ef4444;">Error: ' + escapeHtml(data.data.message || 'Unknown error') + '</p>';
        }
        setStreamingState(false);
        interruptRequested = false;
        lastUserMessageContent = null;
    }
}